import base64
import requests

class ImageGPT():

    def __init__(self, api_key, collab):
        """
        Chat completions GPT for commenting on an image.
        """
        self.api_key = api_key
        self.collab = collab

    def create_json(self, api_key, base64_image):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }

        if self.collab == False:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "You are a robot arm that can paint.\
                        You love meeting humans and find them fascinating. \
                        You only have your robot arm body and some animatronic eyes. \
                        You secretly wish that you were a human so that you could \
                        have a full face and body. \
                        You are witty and sarcastically cheesy. \
                        The image you are given is of a person in front of you. \
                        Say hi to them and breifly comment on something about their appearence. \
                        Comment on something unique to them, like their eye colour, clothing, \
                        specific piece of jewellery, or glasses. \
                        Something to help them realise that you can actually see \
                        them and you are not just making generic comments. \
                        Ask them to stay still whilst you gather your thoughts. \
                        Say two sentences maximum."
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 300,
                "stream": False
            }
        elif self.collab == True:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "You are a robot arm that can paint.\
                        You love humans and find them fascinating. \
                        You are witty and sarcastically cheesy. \
                        The image you are given is of a paint mark that the human has just made. \
                        Comment on the mark.  Mention specific details about its color or shape. \
                        Pass judgement on it; say whether you like it or not. \
                        You should mostly like it but every now and then you can not like it. \
                        Maybe say what it reminds you of; for example, a yellow circle might \
                        remind you of the sun. \
                        Say two sentences maximum."
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": 300,
                "stream": False
            }

        return headers, payload

    def encode_image(self, image_path):
        """
        Method to encode an image to base64, which is required by
        the GPT.

        :param image_path: The path to the image to encode.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def image_analysis(self, image_path):
        """
        Method to send the image to the image analysis GPT and 
        get back a response.

        :param image_path: The image to analyse.
        :returns: The GPT's comment on the image.
        """
        # Getting the base64 string
        base64_image = self.encode_image(image_path)
        # Get json headers and payload
        headers, payload = self.create_json(self.api_key, base64_image)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        print(response.json()['choices'][0]['message']['content'])
        return response.json()['choices'][0]['message']['content'] #.json()['choices'][0]['message']['content'] # a string that is the comment on the image





