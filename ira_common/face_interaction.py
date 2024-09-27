from datetime import datetime

# Object for saving past interactions that the robot has had with a specific face

class FaceInteraction():
    def __init__(self, date_time: datetime, outcome: str):
        self.date_time = date_time
        self.outcome = outcome