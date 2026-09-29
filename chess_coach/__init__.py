"""Chess coach for the web app: game review, principles, openings and the player's style profile.

Built on hobbies/chess (My_folder): its explainer is copied in, and its phase rule, blunder
thresholds and style metrics are carried over. Engine analysis happens in the browser (Stockfish
compiled to JavaScript), so the server needs only python-chess.
"""

from .principles import PRINCIPLES
from .profile import build_profile
from .review import COACH_VERSION, identify_opening, review_game

__all__ = ["COACH_VERSION", "PRINCIPLES", "build_profile", "identify_opening", "review_game"]
