from ira_common.face_interaction import FaceInteraction
from datetime import datetime

class Face():
    def __init__(self, location, size, encoding):
        self.location = location
        self.size = size
        self.encoding = encoding
        self.centred = None
        self.close = None
        self.known = False
        self.past_interactions = []

    def add_interaction(self, date_time: datetime, outcome: str):
        interaction = FaceInteraction(date_time, outcome)
        self.past_interactions.append(interaction)