# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
import json
import os
import random
from contextlib import asynccontextmanager
from copy import deepcopy
from itertools import count
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig, Response,
                               TurbomindEngineConfig)
from lmdeploy.model import MODELS, ChatTemplateConfig, best_match_model
from lmdeploy.serve.utils import LogitsMixin, _get_event_loop
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import _get_and_verify_max_len, _stop_words, get_logger

logger = get_logger('lmdeploy')


def get_names_from_model(model_path: str, model_name: str = None):
    """Get model name and chat template name from workspace model."""
    triton_model_path = os.path.join(model_path, 'triton_models', 'weights')
    if not os.path.exists(triton_model_path):
        chat_template_name = best_match_model(model_path)
    else:
        # `model_path` refers to a turbomind model, reading
        # chat_template_name from the config
        config_path = os.path.join(triton_model_path, 'config.yaml')
        with open(config_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        chat_template_name = config['model_config']['chat_template']
    model_name = model_name if model_name else model_path
    return model_name, chat_template_name


@dataclasses.dataclass
class GenOut:
    """Pack all response information together."""
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Optional[Literal['stop', 'length']] = None
    token_ids: List[int] = None
    logprobs: List[Dict[int, float]] = None


class Session:
    """Session for AsyncEngine.chat.

    Args:
        _id (int): session_id for internal use.
        _step (int): the offset of the k/v cache for internal use.
        _prompt (Any): input prompt for internal use.
        _response (Reaponse): model output for prompt.
        _engine (Any): engine for internal use.
        history (List[Any, str]): chat history.
    """
    _ids = count(0)

    def __init__(self):
        self._id: int = next(self._ids)
        self._step: int = 0
        self._prompt: Any = None
        self._response: Response = None
        self._engine: Any = None
        self.history: List[Tuple[Any, str]] = []

    def _merge_response(self, resp: Response, step: Union[Response, GenOut]):
        """merge response."""
        resp.text += step.text if isinstance(step, Response) else step.response
        resp.input_token_len = step.input_token_len
        resp.generate_token_len = step.generate_token_len
        resp.finish_reason = step.finish_reason
        return resp

    @property
    def response(self) -> Response:
        """return response."""
        return self._response

    def close(self):
        """release engine storage for this session."""
        if self._engine:
            inst = self._engine.create_instance()
            inst.end(self._id)

    def __repr__(self) -> str:
        res = ''
        for user, assistant in self.history:
            if isinstance(user, list):
                user = str(user)
            res += f'USER:\n{user}\nASSISTANT:\n{assistant}\n'
        return res


class AsyncEngine(LogitsMixin):
    """Async inference engine. Maintaining a bunch of tm_model instances.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm/internlm-chat-7b",
            "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat" and so on.
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
    """

    def __init__(self,
                 model_path: str,
                 model_name: Optional[str] = None,
                 backend: Literal['turbomind', 'pytorch'] = 'turbomind',
                 backend_config: Optional[Union[TurbomindEngineConfig,
                                                PytorchEngineConfig]] = None,
                 chat_template_config: Optional[ChatTemplateConfig] = None,
                 **kwargs) -> None:
        logger.info(
            f'input backend={backend}, backend_config={backend_config}')
        logger.info(f'input chat_template_config={chat_template_config}')

        self.model_name, chat_template_name = get_names_from_model(
            model_path, model_name)
        if chat_template_config is None:
            chat_template_config = ChatTemplateConfig(chat_template_name)
        elif chat_template_config.model_name is None:
            chat_template_config.model_name = chat_template_name
        self.chat_template = chat_template_config.chat_template

        logger.info(f'updated chat_template_onfig={chat_template_config}')

        # build backend engine
        if backend == 'turbomind':
            self._build_turbomind(model_path=model_path,
                                  backend_config=backend_config,
                                  **kwargs)
        elif backend == 'pytorch':
            self._build_pytorch(model_path=model_path,
                                backend_config=backend_config,
                                **kwargs)
        else:
            raise ValueError(f'unsupported backend {backend}')

        logger.info(f'updated backend_config={self.backend_config}')

        # parameters for member functions
        self.session_len = _get_and_verify_max_len(
            self.hf_tm_cfg, self.backend_config.session_len)
        self.stop_words = _stop_words(self.chat_template.stop_words,
                                      self.engine.tokenizer)
        if self.stop_words is not None:
            self.stop_words = self.stop_words[0][0].tolist()
        self.backend = backend
        self.instance_num = self.backend_config.max_batch_size
        self.tokenizer = self.engine.tokenizer
        self.id2step = {}
        self.id2generator = {}
        self.running_session_ids = set()
        self.gens_set = set()
        for i in range(self.instance_num):
            self.gens_set.add(self.engine.create_instance())
        self._session_id = count(0)

    def _build_turbomind(
            self,
            model_path: str,
            backend_config: Optional[Union[TurbomindEngineConfig,
                                           PytorchEngineConfig]] = None,
            **kwargs):
        """Innter build method for turbomind backend."""
        from lmdeploy import turbomind as tm
        self.engine = tm.TurboMind.from_pretrained(
            model_path, engine_config=backend_config, **kwargs)
        self.backend_config = self.engine.engine_config
        self.hf_tm_cfg = self.engine.config

    def _build_pytorch(
            self,
            model_path: str,
            backend_config: Optional[Union[TurbomindEngineConfig,
                                           PytorchEngineConfig]] = None,
            **kwargs):
        """Innter build method for pytorch backend."""
        from lmdeploy.pytorch.engine import Engine
        self.engine = Engine(model_path=model_path,
                             engine_config=backend_config)
        self.backend_config = self.engine.engine_config
        self.hf_tm_cfg = getattr(self.engine.model_config, 'hf_config', None)

    def __call__(self,
                 prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
                 gen_config: Optional[GenerationConfig] = None,
                 do_preprocess: bool = True,
                 adapter_name: Optional[str] = None,
                 use_tqdm: bool = False,
                 **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                prompts. It accepts: string prompt, a list of string prompts,
                a chat history in OpenAI format or a list of chat history.
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
            use_tqdm (bool): Whether use the progress bar. Default to False
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        return self.batch_infer(prompts,
                                gen_config=gen_config,
                                do_preprocess=do_preprocess,
                                adapter_name=adapter_name,
                                use_tqdm=use_tqdm,
                                **kwargs)

    async def stop_session(self, session_id: int):
        """Stop a session by a session_id."""
        if str(session_id) in self.id2generator:
            await self.id2generator[str(session_id)].async_cancel(session_id)
            self.gens_set.add(self.id2generator[str(session_id)])

        self.running_session_ids.discard(session_id)

    async def end_session(self, session_id: int):
        """Clear a session by a session_id."""
        if str(session_id) in self.id2generator:
            await self.id2generator[str(session_id)].async_end(session_id)
            self.id2step[str(session_id)] = 0
            self.gens_set.add(self.id2generator[str(session_id)])

        self.running_session_ids.discard(session_id)

    @asynccontextmanager
    async def safe_run(self, session_id: Optional[int] = None):
        """A context manager to make sure server's safe running."""
        try:
            yield
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            # TODO: find out why await would block the coroutine here
            _get_event_loop().create_task(self.stop_session(session_id))
            raise e
        if str(session_id) in self.id2generator:
            self.gens_set.add(self.id2generator[str(session_id)])
        self.running_session_ids.discard(session_id)

    async def get_generator(self, stop: bool, session_id: int):
        """Only return the model instance if it is available."""
        if stop:
            return self.engine.create_instance()
        # waiting no generator is available or the same session_id is running
        while self.gens_set == set() or session_id in self.running_session_ids:
            await asyncio.sleep(0.1)
        generator = self.gens_set.pop()
        self.id2generator[str(session_id)] = generator
        self.running_session_ids.add(session_id)
        return generator

    def batch_infer(self,
                    prompts: Union[List[str], str, List[Dict],
                                   List[List[Dict]]],
                    gen_config: Optional[Union[GenerationConfig,
                                               List[GenerationConfig]]] = None,
                    do_preprocess: bool = True,
                    adapter_name: Optional[str] = None,
                    use_tqdm: bool = False,
                    **kwargs):
        """Inference a batch of prompts.

        Args:
            prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                prompts. It accepts: string prompt, a list of string prompts,
                a chat history in OpenAI format or a list of chat history.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
            use_tqdm (bool): Whether use the progress bar. Default to False
        """
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        if gen_config is None:
            gen_config = GenerationConfig()
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config), \
                'input gen_confg length differs from the length of prompts'  # noqa
        prompt_num = len(prompts)
        session_ids = [next(self._session_id) for _ in range(prompt_num)]
        outputs = [
            Response('', 0, 0, session_ids[i], index=i)
            for i in range(prompt_num)
        ]
        generators = []
        if use_tqdm:
            import tqdm
            pbar = tqdm.tqdm(total=len(prompts))
        for i, prompt in enumerate(prompts):
            generators.append(
                self.generate(prompt,
                              session_ids[i],
                              gen_config=gen_config[i],
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs[i].text += out.response
                outputs[i].generate_token_len = out.generate_token_len
                outputs[i].input_token_len = out.input_token_len
                outputs[i].finish_reason = out.finish_reason
                if out.token_ids:
                    outputs[i].token_ids.extend(out.token_ids)
                if out.logprobs:
                    if outputs[i].logprobs is None:
                        outputs[i].logprobs = []
                    outputs[i].logprobs.extend(out.logprobs)
                if use_tqdm and out.finish_reason is not None:
                    pbar.update(1)

        async def gather():
            await asyncio.gather(
                *[_inner_call(i, generators[i]) for i in range(len(prompts))])

        _get_event_loop().run_until_complete(gather())
        outputs = outputs[0] if need_list_wrap else outputs
        return outputs

    def stream_infer(
            self,
            prompts: Union[List[str], str, List[Dict], List[List[Dict]]],
            gen_config: Optional[Union[GenerationConfig,
                                       List[GenerationConfig]]] = None,
            do_preprocess: bool = True,
            adapter_name: Optional[str] = None,
            **kwargs):
        """Inference a batch of prompts with stream mode.

        Args:
            prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                prompts. It accepts: string prompt, a list of string prompts,
                a chat history in OpenAI format or a list of chat history.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], Dict)
        prompts = [prompts] if need_list_wrap else prompts
        assert isinstance(prompts, List), 'prompts should be a list'
        if gen_config is None:
            gen_config = GenerationConfig()
        if not isinstance(gen_config, List):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config), \
                'input gen_confg length differs from the length of prompts'  # noqa
        session_ids = [next(self._session_id) for _ in range(len(prompts))]
        outputs = Queue()
        generators = []
        for i, prompt in enumerate(prompts):
            generators.append(
                self.generate(prompt,
                              session_ids[i],
                              gen_config=gen_config[i],
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs.put(
                    Response(out.response,
                             out.generate_token_len,
                             out.input_token_len,
                             session_ids[i],
                             out.finish_reason,
                             out.token_ids,
                             out.logprobs,
                             index=i))

        async def gather():
            await asyncio.gather(
                *[_inner_call(i, generators[i]) for i in range(len(prompts))])
            outputs.put(None)

        loop = _get_event_loop()
        proc = Thread(target=lambda: loop.run_until_complete(gather()))
        proc.start()

        while True:
            try:
                out = outputs.get(timeout=0.001)
                if out is None:
                    break
                yield out
            except Empty:
                pass

        proc.join()

    async def _get_prompt_input(self,
                                prompt: str,
                                do_preprocess: bool,
                                sequence_start: bool,
                                adapter_name: str,
                                tools: Optional[List[object]] = None,
                                **kwargs):
        # print(f"__file__ is {__file__} set do preprocess is False")
        do_preprocess = False 
        if do_preprocess:
            # use adapter's chat template if possible
            chat_template = self.chat_template
            if adapter_name in MODELS.module_dict:
                chat_template = MODELS.module_dict[adapter_name]()
            prompt = chat_template.messages2prompt(prompt,
                                                   sequence_start,
                                                   tools=tools)
        input_ids = self.tokenizer.encode(prompt, add_bos=sequence_start)
        return {'prompt': prompt, 'input_ids': input_ids}

    async def generate(
            self,
            messages,
            session_id: int,
            gen_config: Optional[GenerationConfig] = None,
            tools: Optional[List[object]] = None,
            stream_response: bool = True,
            sequence_start: bool = True,
            sequence_end: bool = True,  # no interactive mode by default
            step: int = 0,
            do_preprocess: bool = True,
            adapter_name: Optional[str] = None,
            **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether return responses streamingly
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """
        if str(session_id) not in self.id2step:
            self.id2step[str(session_id)] = 0
        if step != 0:
            self.id2step[str(session_id)] = step
        if gen_config is None:
            gen_config = GenerationConfig()
        else:
            gen_config = deepcopy(gen_config)
        gen_config.convert_stop_bad_words_to_ids(self.tokenizer)
        if gen_config.stop_token_ids is None:
            gen_config.stop_token_ids = self.stop_words
        if not gen_config.do_sample:
            # greedy decode
            gen_config.top_k = 1
            # avoid unnecessary process
            gen_config.temperature = 1.0
            gen_config.repetition_penalty = 1.0
        # set random if it is not set and sequence_start is True
        if gen_config.random_seed is None and sequence_start:
            gen_config.random_seed = random.getrandbits(64)
        if gen_config.n > 1:
            logger.warning(f"n({gen_config.n}) > 1 hasn't been supported yet. "
                           f'Fallback to 1')
            gen_config.n = 1
        prompt = messages

        prompt_input = await self._get_prompt_input(prompt,
                                                    do_preprocess,
                                                    sequence_start,
                                                    adapter_name,
                                                    tools=tools)
        prompt = prompt_input['prompt']
        input_ids = prompt_input['input_ids']
        finish_reason = None
        logger.info(f'prompt={prompt!r}, '
                    f'gen_config={gen_config}, '
                    f'prompt_token_id={input_ids}, '
                    f'adapter_name={adapter_name}.')
        logger.info(f'session_id={session_id}, '
                    f'history_tokens={self.id2step[str(session_id)]}, '
                    f'input_tokens={len(input_ids)}, '
                    f'max_new_tokens={gen_config.max_new_tokens}, '
                    f'seq_start={sequence_start}, seq_end={sequence_end}, '
                    f'step={step}, prep={do_preprocess}')

        if gen_config.max_new_tokens is None:
            # for interactive endpoint, will try maximum possible token num
            gen_config.max_new_tokens = max(
                128, self.session_len - self.id2step[str(session_id)] -
                len(input_ids))
        elif self.id2step[str(session_id)] + len(
                input_ids) + gen_config.max_new_tokens > self.session_len:
            gen_config.max_new_tokens = max(
                self.session_len - self.id2step[str(session_id)] -
                len(input_ids), 128)
            logger.error(
                f'Truncate max_new_tokens to {gen_config.max_new_tokens}')
        if self.id2step[str(session_id)] + len(
                input_ids) + gen_config.max_new_tokens > self.session_len:
            logger.error(f'run out of tokens. session_id={session_id}.')
            yield GenOut('', self.id2step[str(session_id)], len(input_ids), 0,
                         'length')
            if sequence_end is True and sequence_start is False:
                await self.end_session(session_id)
        else:
            generator = await self.get_generator(False, session_id)
            async with self.safe_run(session_id):
                state = DetokenizeState(len(input_ids))
                start_ids_offset = state.ids_offset
                response = ''
                async for outputs in generator.async_stream_infer(
                        session_id=session_id,
                        **prompt_input,
                        gen_config=gen_config,
                        adapter_name=adapter_name,
                        stream_output=stream_response,
                        sequence_start=sequence_start,
                        sequence_end=sequence_end,
                        step=self.id2step[str(session_id)]):
                    # decode res
                    res, tokens = input_ids + outputs.token_ids, outputs.num_token  # noqa
                    if len(res) <= state.ids_offset:
                        continue

                    ids_offset = state.ids_offset
                    response, state = self.tokenizer.detokenize_incrementally(
                        res,
                        state,
                        skip_special_tokens=gen_config.skip_special_tokens)

                    res = res[ids_offset:]
                    logprobs = None
                    if outputs.logprobs:
                        log_offset = ids_offset - start_ids_offset
                        logprobs = outputs.logprobs[log_offset:]

                    # response, history token len,
                    # input token len, gen token len
                    yield GenOut(response, self.id2step[str(session_id)],
                                 len(input_ids), tokens, finish_reason, res,
                                 logprobs)

                finish_reason = 'length' \
                    if tokens >= gen_config.max_new_tokens else 'stop'
                # utf-8 char at the end means it's a potential unfinished
                # byte sequence
                if not response.endswith('�'):
                    response = ''  # avaid returning the last response twice
                yield GenOut(response, self.id2step[str(session_id)],
                             len(input_ids), tokens, finish_reason)
                # update step
                self.id2step[str(session_id)] += len(input_ids) + tokens
                if sequence_end:
                    self.id2step[str(session_id)] = 0
                # manually end pytorch session
                # TODO modify pytorch or turbomind api
                if self.backend == 'pytorch' and sequence_end:
                    await self.end_session(session_id)

    def parse_tool_response(self, text, tools, **kwargs):
        """Parse model response containing tool information.

        Args:
            text(str): model response in string format
            tools(List): tools from user request
        """
        if '<|plugin|>' in text:  # internlm2
            text, action = text.split('<|action_start|><|plugin|>')
            action = action.split('<|action_end|>'.strip())[0]
            action = action[action.find('{'):]
            action = json.loads(action)
            name, parameters = action['name'], json.dumps(
                action.get('parameters', action.get('arguments', {})))
        elif '<function=' in text:  # llama3.1
            action, _ = text.split('</function>')
            parameters = action[action.find('{'):]
            name = action.split('<function=')[1].split('>{')[0]
        else:
            raise RuntimeError(f'Unexpected model response: {text}')
        action_id = [tool.function.name for tool in tools].index(name)
        return text, action_id, name, parameters

    def chat(self,
             prompt: str,
             session=None,
             gen_config: Optional[GenerationConfig] = None,
             do_preprocess: bool = True,
             **kwargs) -> Session:
        """Chat.

        Args:
            prompt (str): prompt
            session (Session): the chat session
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            **kwargs (dict): ad hoc parametrization of `gen_config
        """
        if session is None:
            session = Session()
            session._engine = self.engine

        # sync & init
        session._prompt = prompt
        session._response = None

        sequence_start = session._step == 0

        async def _work():
            resp = Response('', -1, -1, session._id)
            async for output in self.generate(prompt,
                                              session_id=session._id,
                                              gen_config=gen_config,
                                              stream_response=False,
                                              sequence_start=sequence_start,
                                              sequence_end=False,
                                              step=session._step,
                                              do_preprocess=do_preprocess,
                                              **kwargs):
                resp = session._merge_response(resp, output)
            return resp

        from lmdeploy.pytorch.engine.request import _run_until_complete
        resp = _run_until_complete(_work())

        session._response = resp
        session._step += resp.generate_token_len + resp.input_token_len
        session.history.append((session._prompt, resp.text))

        return session
