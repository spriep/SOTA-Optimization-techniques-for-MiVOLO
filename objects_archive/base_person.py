from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Person:
    """Class to store all the state of a person in a frame."""
    # Tracking ID 
    id: int 
    
    # Body coordinates and confidence score (RT-DETR)
    body_box: List[int]      # [x1, y1, x2, y2]
    body_score: float
    
    # Face coordinates and confidence score (RetinaFace)
    face_box: Optional[List[int]] = None
    face_score: Optional[float] = None
    
    # Output from MiVOLO
    age: Optional[float] = None
    #gender: Optional[str] = None           # Not implementing gender recognition for now, but we could add it easily in the future if we want

    def has_face(self) -> bool:
        """Returns True if this person has a face associated in this frame."""
        return self.face_box is not None

    def get_face_center(self) -> Optional[tuple]:
        """Calculates the center of the face."""
        if not self.has_face():
            return None
        fx1, fy1, fx2, fy2 = self.face_box
        return ((fx1 + fx2) // 2, (fy1 + fy2) // 2)