# This class file is for Single Prompt -> Single Response chat with the LLM taking advantage of the HumanMessage class, where a chat is invoked with a single message
from langchain_core.messages import HumanMessage, SystemMessage
from src.model.wrappers import ChatModelWrapper, EmbeddingWrapper
from src.model.model_catalogue import ModelType, EmbeddingType, Providers, ModelCatalogue
from src.data.attachments import AttachmentTypes, Attachment
from src.data.model_format_exception import ModelFormatException
# Import prompttemplate from langchain_core
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from json import JSONEncoder, JSONDecoder
from typing import Tuple

class ChatMessageGenerator:
    def __init__(self, model: ChatModelWrapper,system_prompt : str):
        self.model = model
        self.system_prompt = system_prompt
        self.chat = [("system", system_prompt)]


    def prepResponse(self,  attachments: list = None) -> Tuple[str, None]:
        model_type = self.model.model_type
        modality = model_type.multiModal and (attachments is not None and len(attachments) > 0)
        if modality:
            return ("user", ChatMessageGenerator.provider_multimodal_generator(model_type, attachments))
        if not modality:
            return ("user", "{prompt}")
        

    def provider_multimodal_generator(modelType: ModelType, attachments: list, attachment_constant: str = "attachment"):
        provider = modelType.provider
        content = [{"type": "text", "text": "{prompt}"}]
        for i, attachment in enumerate(attachments):
            attachment.prompt_mapping = f"{attachment_constant}{i}"
            if attachment.attachment_type == AttachmentTypes.IMAGE:
                content.append({
                    "type": "image_url",
                    "image_url" : {
                        "url" : "data:image/jpeg;base64,{" + attachment.prompt_mapping + "}"
                    }
                })
            elif attachment.attachment_type == AttachmentTypes.AUDIO:
                content.append({
                    "type": "input_audio",
                    "input_audio" : {
                        "data" : "{" + attachment.prompt_mapping + "}"
                    }
                })
            elif attachment.attachment_type == AttachmentTypes.VIDEO:
                raise NotImplementedError("Video attachments are not yet supported")
            elif attachment.attachment_type == AttachmentTypes.FILE:
                raise NotImplementedError("File attachments are not yet supported")
            else:
                raise ModelFormatException(modelType.argName, modelType.multiModal, attachment, provider, "Unsupported attachment type")
        return content

