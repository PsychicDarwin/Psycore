# This class file is for Single Prompt -> Single Response chat with the LLM taking advantage of the HumanMessage class, where a chat is invoked with a single message
from langchain_core.messages import HumanMessage, SystemMessage
from src.model.wrappers import ModelWrapper, EmbeddingWrapper
from src.model.model_catalogue import ModelType, EmbeddingType, Providers
from src.data.attachments import AttachmentTypes
# Import prompttemplate from langchain_core
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from json import JSONEncoder, JSONDecoder

class MessageGenerator:
    def __init__(self, model: ModelWrapper, embedding: EmbeddingWrapper,system_prompt : str):
        self.model = model
        self.embedding = embedding
        self.system_prompt = system_prompt


    def prepResponse(self,  attachments: list = None):
        modality = self.model.model_type.multiModal and (attachments is not None and len(attachments) > 0)
        if modality:
            if Providers.OPENAI == self.model.model_type.provider:
                return ChatPromptTemplate([
                    ("system", self.system_prompt),
                    ("user", [
                        {
                            "type": "text",
                            "text": 
                        },
                        {

                        }
                    ])
                ])
            elif Providers.BEDROCK == self.model.model_type.provider:
                pass
            elif Providers.GEMINI == self.model.model_type.provider:
                pass
            elif Providers.OLLAMA == self.model.model_type.provider:
                pass
            else:
                modality = False
        if not modality:
            return ChatPromptTemplate([
                ("system", self.system_prompt),
                ("user", "{prompt}")
            ])
        

    def provider_multimodal_generator(provider: Providers, attachments: list, attachment_constant: str = "attachment"):
        if Providers.OPENAI == provider:
            content = [{"type": "text", "text": "{prompt}"}]
            for i, attachment in enumerate(attachments):
                attachment.prompt_mapping = f"{attachment_constant}{i}"
                if attachment.attachment_type == AttachmentTypes.IMAGE:
                    content.append({
                        "type": "image_url",
                        "image_url" : {
                            "url" : "data:image/jpeg;base64,\{{0}}\}".format(attachment.prompt_mapping)
                        }
                    })
                elif attachment.attachment_type == AttachmentTypes.AUDIO:
                    content.append({
                        "type": "input_audio",
                        "input_audio" : {
                            "data" : "\{{0}}\}".format(attachment.prompt_mapping)
                        }
                    })
                elif attachment.attachment_type == AttachmentTypes.VIDEO:
                    raise NotImplementedError("Video attachments are not yet supported")
