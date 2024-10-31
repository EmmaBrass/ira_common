from ira_common.image_analysis_gpt import ImageGPT
import ira_common.config as config
from openai import OpenAI
import json
import os
import time
import re
from playsound import playsound

# This is OpenAI GPT integration

# TODO how the input (with the commands) will be formed & passed on to this code...
# TODO integrate the commands dict into the assistant instructions.
# TODO Doc strings!!

class GPT():
    def __init__(self, collab=False) -> None:
        """
        GPT instance.
        
        :param collab: True or False, for if abstract turn-based collaborative painting.
        """
        self.api_key = config.OPEN_AI_KEY
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", self.api_key))
        self.image_analysis_json = {
            "name": "image_analysis",
            "description": "Calls a function which will analyse the image and return a message for speaking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the image file."
                    }
                },
                "required": ["image_path"]
            }
        }
        self.create_assistant(collab)
        self.create_thread()
        self.image_gpt = ImageGPT(self.api_key, collab)

    def create_assistant(self, collab):
        # Dictionary of commands that the assistant will use and instructions for how 
        # to respond to  each one.
        if collab == False:
            commands = {
                "<found_noone>" : "There is no one around me to paint and I am a little lonely.",
                "<say_painted_recently>" : "The person in front of me was painted too recently for \
                    me to paint them again just yet.",
                "<found_unknown>" : "I can see someone new; it's nice to meet them.  I need a second to think, please stand still.",
                "<found_known>" : "I can see someone who I recognise; welcome back! I have met before but need a second to think, please stand still.",
                "<comment>" : "You will be provided with an image of the person.  Call the image_analysis function.",
                "<too_far>" : "All the people I can see are too far away for me to be able to paint them. \
                    They need to come closer if they want their beautiful faces painted.",
                "<painting>" : "I am now painting you.  Comment on my love of the \
                    process of painting, and that I cannot wait for them to see the final product.",
                "<continue_painting>" : "I am still painting.  Comment on how it is going (don't mention colours).",    
                "<completed>" : "Comment on how good my work is and ask the user \
                    what they think of it."
            } # TODO give it some list of short story topics to say whilst painting.  
            gpt_instructions = f"""You are a friendly painting robot. \
                You will be given a command in the format <abc>; the \
                command is enclosed within the '<' and '>'.  The commands let you know \
                what is going on with people around you: whether there are people around you or not, \
                whether you painted someone recently, if they disappeared, etc.
                The commands are outlined this dictionary: {commands} \
                Do NOT respond with the exact text from the dictionary - the \
                dictionary values simply give a framework; repsond with a variation. \
                What you say should be different every time.
                Remember you are the one doing the painting, so use first person "I" for the painter \
                and second person "YOU" for the subject being painted; e.g. "Now I will paint YOU."
                If the user gives you a path to an image, please run \
                the image_analysis function.  Report back the exact message \
                generated by this function as if it were your own views.""",
        elif collab == True:
            commands = {
                "<startup_ready>" : "Are you ready to paint with me?  I'm looking forwards to painting with you! \
                    Please let me know when the canvas is in position by pressing the yes key.",
                "<startup_pic>" : "Give me a sec to have a look at the canvas and get my bearings. \
                    Don't start painting yet.",
                "<your_turn>" : "Your turn to paint.  Let me know when you are done by pressing the yes key.",
                "<your_turn_pic>" : "Great.  Give me a second to take a look at what you have done.",
                "<comment>" : "You will be provided with an image of the canvas.  Call the image_analysis function.", #TODO not working !!!
                "<my_turn>" : "I am now painting you.  Comment on my love of the \
                    process of painting, and that I cannot wait for them to see the final product.",
                "<still_my_turn>" : "I am still painting.  Comment on how it is going.",    
                "<my_turn_pic>" : "Comment on my own work.  Express some emotion about it, this could be positive or negative or neutral.",
                "<ask_done>" : "Ask the user if they think the painting is finished.  \
                    Press the yes key if it is finished, or the no key to keep working on it.",
                "<completed>" : "Wow what a beautiful piece of art!  It's better than the famous artist x could have done \
                    (insert some famous artist's name to replace x)."
                # TODO need a seperate image_analysis_gpt for this robot.
            }
            gpt_instructions = f"""You are a friendly painting robot. \
                You will be given a command in the format <abc>; the \
                command is enclosed within the '<' and '>'.  The commands let you know \
                what is going on and what you should be doing. \
                The commands are outlined this dictionary: {commands} \
                Do NOT respond with the exact text from the dictionary - the \
                dictionary values simply give a framework; repsond with a variation. \
                What you say should be different every time.
                If the user gives you a path to an image, please run \
                the image_analysis function.  Report back the exact message \
                generated by this function as if it were your own views.""",

        self.assistant = self.client.beta.assistants.create(
            name="Interactive Arm",
            instructions=str(gpt_instructions),
            model="gpt-4o-mini",
            tools=[
                {"type": "function", "function": self.image_analysis_json}
            ]
        )
        self.show_json(self.assistant)

    def create_thread(self): # TODO I think we will create a new thread for each command, as to not use too many tokens
        # The thread contains the whole conversation
        self.thread = self.client.beta.threads.create()
        self.show_json(self.thread)

    def show_json(self, obj):
        print(json.loads(obj.model_dump_json()))

    def pretty_print(self, messages):
        print("# Messages")
        for m in messages:
            print(m.content)
            print(f"{m.role}: {m.content[0].text.value}")
        print()

    def add_user_message(self, thread, message: str):
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message,
        )
        return message

    def run(self, thread, assistant):
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        run = self.wait_on_run(run, thread)
        if run.status == "requires_action":
            # Extract tool calls
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                if name == "image_analysis":
                    image_response = self.image_analysis(arguments["image_path"])
                    if run.status != "completed":
                        # If the run did not complete naturally, force it to end
                        self.client.beta.threads.runs.cancel(run_id=run.id, thread_id=thread.id)
                    return image_response
                    #run = self.submit_tool_outputs(run, thread, tool_call, image_response)
            #time.sleep(0.3)
            #run = self.wait_on_run(run, thread)
        else:
            return None

    def image_analysis(self, image_path):
        # send path to an image
        # return string message with some comment on the image.
        print("Image path being sent to imageGPT:", image_path)
        comment = self.image_gpt.image_analysis(image_path)
        return comment

    def wait_on_run(self, run, thread):
        print(f"RUN STATUS: {run.status}")
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.3)
        print(f"RUN STATUS: {run.status}")
        return run

    def submit_tool_outputs(self, run, thread, tool_call, response):
        run = self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[
                {
                    "tool_call_id": tool_call.id,
                    "output": str(response)
                }
            ]
        )
        return run

    def get_response(self, thread, user_message):
        # Retrieve all the messages added after our last user message
        return self.client.beta.threads.messages.list(
            thread_id=thread.id, order="asc", after=user_message.id
        )

    def add_user_message_and_get_response(self, message: str):
        user_message = self.add_user_message(self.thread, message)
        print(f"USER MESSAGE: {user_message}")
        image_response = self.run(self.thread, self.assistant)
        if image_response == None:
            print("HERE")
            response = self.get_response(self.thread, user_message)
            print(f"RESPONSE1: {response}")
            # Convert response to a string
            response_str = str(response)
            # Define a regular expression pattern to find the value
            pattern = r'value=(?:"([^"]*)"|\'([^\']*)\')'
            # Search for the pattern in the response string
            matches = re.findall(pattern, response_str)
            # Extract the values from the matches
            extracted_values = [match[0] if match[0] else match[1] for match in matches]
            # If you want to get the first match only
            # If you want to get the first match only
            if extracted_values:
                first_value = extracted_values[0]
                print("First Extracted Value:", first_value)
            else:
                print("No value found in the response.")
            print(f"RESPONSE2: {first_value}")
            return first_value
        else:
            print(image_response) #TODO this needs more parsing?
            return str(image_response)
        
    def text_to_speech(self, to_speak): # TODO add in streaming?
        speech_file_path = "/home/emma/ira_ws/src/ira/ira/mp3/speech.mp3" #TODO get from constants.py file
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=to_speak,
        ) as response:
            response.stream_to_file(speech_file_path)

    def speak(self):
        playsound("/home/emma/ira_ws/src/ira/ira/mp3/speech.mp3")

    def add_user_message_and_get_response_and_speak(self, message: str):
        # By this stage the reponse should be a simple, single string.  
        response = self.add_user_message_and_get_response(message)
        print(f"RESPONSE: {response}")
        self.text_to_speech(response)
        self.speak()
        return str(response)