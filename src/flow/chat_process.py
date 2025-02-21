from src.flow.processflow import ProcessFlow
from src.model.wrappers import ChatModelWrapper
from src.interaction.message import ChatMessageGenerator
from src.data.attachments import AttachmentTypes, Attachment
from langchain_core.prompts import ChatPromptTemplate
from src.model.model_catalogue import ModelCatalogue

class ChatProcess(ProcessFlow):
    def __init__(self, system_prompt: str = None, type: str = None, next: 'ProcessFlow' = None):
        super().__init__(ChatMessageGenerator(
            ChatModelWrapper(
                ModelCatalogue._models[type]
            ),
            system_prompt
        ), next)
        

    def run(self,data : dict) -> dict:
        # When running a chat process we expect data in the form
        # {
        #     "attachments" : list, (list of strings)
        #     "prompt" : str,
        #     "append" : bool   # Decides whether to append the response and original message to the chat
        # }

        # We extract all attachments from the data

        attachments = data.get("attachments",[])
        attachment_objects = [Attachment(AttachmentTypes.from_filename(attachment),attachment,True) for attachment in attachments]
        Attachment.extractAttachmentList(attachment_objects)
        
        prep_attachments = self.runner.prepResponse(attachments=attachment_objects)
        self.runner.chat.append(prep_attachments)


        self.chat_template = ChatPromptTemplate(self.runner.chat)


        chain_input = Attachment.attachmentListMapping(attachment_objects)
        chain_input.update({"prompt" : data["prompt"], })


        output = self.chat(chain_input)

        if data.get("append",False):
            chat_output = self.chat_template.invoke(chain_input)
            # Replace the last message with the chat output, as it originaly had placeholder values
            self.runner.chat[-1] = ("user",chat_output.messages.pop().content)
            # Add self.chat output from earlier to chat as system message
            self.runner.chat.append(("ai",output.content)) 
        
        
        if self.next is not None:
            return self.next.run(output)
        return output
    

    def chat(self, langchain_dict : dict) -> dict:
        chain = self.chat_template | self.runner.model.model
        response = chain.invoke(langchain_dict)
        return response
