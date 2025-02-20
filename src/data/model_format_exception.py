from src.data.attachments import AttachmentTypes, Attachment
class ModelFormatException(Exception):
    def __init__(self, model : str, multimodal : bool, attachment : Attachment, provider : str, message : str):
        super().__init__("""
There was an error with the model's usage of Multimodal Data
Provider: {0}
Model Name: {1}
Multimodal: {2}
Attachment type: {3}

Error: {4}
        """.format(provider, model, multimodal, attachment.attachment_type.name, message))
        self.model = model
        self.multimodal = multimodal
        self.attachment = attachment
        self.provider = provider