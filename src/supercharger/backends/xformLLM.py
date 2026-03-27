from supercharger.llm import LLM
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoConfig, TextIteratorStreamer
from threading import Thread
from typing import override
import logging

# L1JSON/redJSON -> xformLLM inflation:
#  params
#   .model       -> model                     load parameters.
#   .processor   -> processor/tokenizer       load parameters.
#  prompt
#   .model       -> model                     generation parameters.
#   .processor   -> processor/tokenizer       data process parameters.
#  data          -> model                     generation messages.

logger = logging.getLogger("lib.xformLLM")

class xformLLM(LLM):
    def __init__(self, json_file: str):
        logger.info("initializing instance...")
        super().__init__(json_file)
        self._load_data()                                                           ; logger.debug("parsed L1JSON succesfully.")
        mtype = self._load_model()                                                  ; logger.debug(f"loaded model ({mtype} type).")

        self.processor = AutoProcessor.from_pretrained(**self.params_processor)     ; logger.debug("set up processor.")
        self.tokenizer = self.processor.tokenizer if mtype != "CAUSAL" else self.processor._tokenizer

    @override
    def callback(self):
        processed = self.processor.apply_chat_template(
            self.data.get("messages", []),
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **self.prompt_processor,
        )
        processed = {k: v.to(self.model.device) for k,v in processed.items()}

        self.streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=False,
            skip_special_tokens=False,
        )
        thread = Thread(
            target=self.model.generate,
            kwargs={**processed, **self.prompt_model, "streamer": self.streamer},
        )
        thread.start()

        logger.info("output ahead!")
        logger.info("%TRLog%_wild_output_begin")
        output = []
        for token in self.streamer:
            output.append(token)
            print(token, end="", flush=True)
        print()
        logger.info("%TRLog%_wild_output_end")
        logger.info("end of output.")

        return output

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.params_model.get("pretrained_model_name_or_path", {}))

        # non-working quantization implementation. use as reference.
        """
        qargs = self.params_quant.get("config", {})
        match self.params_quant.get("type", {}):
            case "SINQ":
                qconf = SinqConfig(**qargs)
            case {}:
                qconf = {}
            case _:
                sys.exit("bad quant type.")
        qconf = {"quantization_config": qconf}
        """

        if hasattr(config, "vision_config"):
            logger.debug("model has vision_config attribute.")
            self.model = AutoModelForMultimodalLM.from_pretrained(**self.params_model)#, **qconf)
            return "MMODAL"

        self.model = AutoModelForCausalLM.from_pretrained(**self.params_model)#, **qconf)
        return "CAUSAL"

    def _load_data(self):
        super()._load_data()

        # PPD -> P1
        self.params_model =     self.params.get("model", {})
        self.params_processor = self.params.get("processor", {})
        # PPD -> P2
        self.prompt_model =     self.prompt.get("model", {})
        self.prompt_processor = self.prompt.get("processor", {})
        # PPD -> D
        # (no processing required.)
