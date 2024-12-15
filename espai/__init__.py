"""Top-level package for Analysis of expert survey on progress in AI."""
import enum
import logging
from pathlib import Path

from diskcache import Cache


PROJECT_ROOT = Path(__file__).parent.parent
cache = Cache(str(PROJECT_ROOT / ".cache"))
project_logger = logging.getLogger("espai")


class Question(enum.Enum):
    HLMI = "HLMI"
    FAOL = "FAOL"

    # TODO: Our method to aggregate HLMI and FAOL is very hacky. See preprocessing code.
    HLMI_FAOL = "HLMI_FAOL"

    TRUCK_DRIVER = "Truck driver"
    SURGEON = "Surgeon"
    RETAIL_SALESPERSON = "Retail salesperson"
    AI_RESEARCHER = "AI researcher"
    FINAL_OCCUPATION = "Final occupation"
    TRANSLATE_NEW_LANGUAGE = "Translate new language"
    TRANSLATE_SPEECH_FROM_SUBTITLES = "Translate speech from subtitles"
    HUMAN_LEVEL_TRANSLATION = "Human-level translation"
    PHONE_BANKING = "Phone banking"
    IMAGE_CLASSIFICATION = "Image classification"
    ONE_SHOT_LEARNING = "One-shot learning"
    THREE_D_MODELING_FROM_VIDEO = "3D modeling from video"
    SPEECH_TRANSCRIPTION = "Speech transcription"
    TEXT_TO_SPEECH = "Text-to-speech"
    MATH_RESEARCH = "Math research"
    PUTNAM_COMPETITION = "Putnam competition"
    GO = "Go"
    STARCRAFT = "Starcraft"
    GENERAL_GAME_PLAYING = "General game playing"
    ANGRY_BIRDS = "Angry Birds"
    ATARI_GAMES = "Atari games"
    ATARI_GAMES_NOVICE = "Atari games novice"
    LAUNDRY_FOLDING = "Laundry folding"
    BIPEDAL_RUNNING = "Bipedal running"
    LEGO_ASSEMBLY = "LEGO assembly"
    SORTING_ALGORITHM = "Sorting algorithm"
    CODE_GENERATION = "Code generation"
    FACTOID_QA = "Factoid QA"
    OPEN_ENDED_QA = "Open-ended QA"
    AMBIGUOUS_QA = "Ambiguous QA"
    HIGH_SCHOOL_ESSAY = "High school essay"
    POP_SONG_COMPOSITION = "Pop song composition"
    ARTIST_SPECIFIC_SONG = "Artist-specific song"
    BESTSELLER_NOVEL = "Bestseller novel"
    GAME_STRATEGY_EXPLANATION = "Game strategy explanation"
    POKER = "Poker"
    PHYSICS_EQUATIONS_DISCOVERY = "Physics equations discovery"
    ELECTRICAL_WIRING = "Electrical wiring"
    LLM_FINE_TUNING = "LLM fine-tuning"
    MATH_PROBLEM_SOLVING = "Math problem solving"
    ML_STUDY_REPLICATION = "ML study replication"
    WEBSITE_CREATION = "Website creation"
    SECURITY_FLAW_DETECTION = "Security flaw detection"


class Framing(enum.Enum):
    FY = "Fixed years"  # column name as it appears in the data
    FP = "Fixed probabilities"  # column name as it appears in the data

    @property
    def human_columns(self):
        match self:
            case Framing.FY:
                return ["p1", "p2", "p3"]
            case Framing.FP:
                return ["x1", "x2", "x3"]
            case _:
                raise ValueError(f"Unknown framing: {self}")


FIELDS = {
    Question.HLMI: {
        Framing.FY: {
            10: "hb_b_1",
            20: "hb_b_2",
            # Note: this asks for 40 years, all other questions ask for 50 years
            40: "hb_b_3",
        },
        Framing.FP: {
            0.1: "hb_a_1",
            0.5: "hb_a_2",
            0.9: "hb_a_3",
        },
    },
    Question.FAOL: {
        Framing.FY: {
            10: "hj_b_full_1",
            20: "hj_b_full_2",
            50: "hj_b_full_3",
        },
        Framing.FP: {
            0.1: "hj_a_full_1",
            0.5: "hj_a_full_2",
            0.9: "hj_a_full_3",
        },
    },
    Question.TRUCK_DRIVER: {
        Framing.FY: {
            10: "hj_b_jobs_1_1",
            20: "hj_b_jobs_1_2",
            50: "hj_b_jobs_1_3",
        },
        Framing.FP: {
            0.1: "hj_a_jobs_1_1",
            0.5: "hj_a_jobs_1_2",
            0.9: "hj_a_jobs_1_3",
        },
    },
    Question.SURGEON: {
        Framing.FY: {
            10: "hj_b_jobs_2_1",
            20: "hj_b_jobs_2_2",
            50: "hj_b_jobs_2_3",
        },
        Framing.FP: {
            0.1: "hj_a_jobs_2_1",
            0.5: "hj_a_jobs_2_2",
            0.9: "hj_a_jobs_2_3",
        },
    },
    Question.RETAIL_SALESPERSON: {
        Framing.FY: {
            10: "hj_b_jobs_3_1",
            20: "hj_b_jobs_3_2",
            50: "hj_b_jobs_3_3",
        },
        Framing.FP: {
            0.1: "hj_a_jobs_3_1",
            0.5: "hj_a_jobs_3_2",
            0.9: "hj_a_jobs_3_3",
        },
    },
    Question.AI_RESEARCHER: {
        Framing.FY: {
            10: "hj_b_jobs_4_1",
            20: "hj_b_jobs_4_2",
            50: "hj_b_jobs_4_3",
        },
        Framing.FP: {
            0.1: "hj_a_jobs_4_1",
            0.5: "hj_a_jobs_4_2",
            0.9: "hj_a_jobs_4_3",
        },
    },
    Question.FINAL_OCCUPATION: {
        Framing.FY: {
            10: "hj_b_final_pred_1",
            20: "hj_b_final_pred_2",
            50: "hj_b_final_pred_3",
        },
        Framing.FP: {
            0.1: "hj_a_final_pred_1",
            0.5: "hj_a_final_pred_2",
            0.9: "hj_a_final_pred_3",
        },
    },
    Question.TRANSLATE_NEW_LANGUAGE: {
        Framing.FY: {
            10: "tb_1_1",
            20: "tb_1_2",
            50: "tb_1_3",
        },
        Framing.FP: {
            0.1: "ta_1_1",
            0.5: "ta_1_2",
            0.9: "ta_1_3",
        },
    },
    Question.TRANSLATE_SPEECH_FROM_SUBTITLES: {
        Framing.FY: {
            10: "tb_2_1",
            20: "tb_2_2",
            50: "tb_2_3",
        },
        Framing.FP: {
            0.1: "ta_2_1",
            0.5: "ta_2_2",
            0.9: "ta_2_3",
        },
    },
    Question.HUMAN_LEVEL_TRANSLATION: {
        Framing.FY: {
            10: "tb_3_1",
            20: "tb_3_2",
            50: "tb_3_3",
        },
        Framing.FP: {
            0.1: "ta_3_1",
            0.5: "ta_3_2",
            0.9: "ta_3_3",
        },
    },
    Question.PHONE_BANKING: {
        Framing.FY: {
            10: "tb_4_1",
            20: "tb_4_2",
            50: "tb_4_3",
        },
        Framing.FP: {
            0.1: "ta_4_1",
            0.5: "ta_4_2",
            0.9: "ta_4_3",
        },
    },
    Question.IMAGE_CLASSIFICATION: {
        Framing.FY: {
            10: "tb_5_1",
            20: "tb_5_2",
            50: "tb_5_3",
        },
        Framing.FP: {
            0.1: "ta_5_1",
            0.5: "ta_5_2",
            0.9: "ta_5_3",
        },
    },
    Question.ONE_SHOT_LEARNING: {
        Framing.FY: {
            10: "tb_6_1",
            20: "tb_6_2",
            50: "tb_6_3",
        },
        Framing.FP: {
            0.1: "ta_6_1",
            0.5: "ta_6_2",
            0.9: "ta_6_3",
        },
    },
    Question.THREE_D_MODELING_FROM_VIDEO: {
        Framing.FY: {
            10: "tb_7_1",
            20: "tb_7_2",
            50: "tb_7_3",
        },
        Framing.FP: {
            0.1: "ta_7_1",
            0.5: "ta_7_2",
            0.9: "ta_7_3",
        },
    },
    Question.SPEECH_TRANSCRIPTION: {
        Framing.FY: {
            10: "tb_8_1",
            20: "tb_8_2",
            50: "tb_8_3",
        },
        Framing.FP: {
            0.1: "ta_8_1",
            0.5: "ta_8_2",
            0.9: "ta_8_3",
        },
    },
    Question.TEXT_TO_SPEECH: {
        Framing.FY: {
            10: "tb_9_1",
            20: "tb_9_2",
            50: "tb_9_3",
        },
        Framing.FP: {
            0.1: "ta_9_1",
            0.5: "ta_9_2",
            0.9: "ta_9_3",
        },
    },
    Question.MATH_RESEARCH: {
        Framing.FY: {
            10: "tb_10_1",
            20: "tb_10_2",
            50: "tb_10_3",
        },
        Framing.FP: {
            0.1: "ta_10_1",
            0.5: "ta_10_2",
            0.9: "ta_10_3",
        },
    },
    Question.PUTNAM_COMPETITION: {
        Framing.FY: {
            10: "tb_11_1",
            20: "tb_11_2",
            50: "tb_11_3",
        },
        Framing.FP: {
            0.1: "ta_11_1",
            0.5: "ta_11_2",
            0.9: "ta_11_3",
        },
    },
    Question.GO: {
        Framing.FY: {
            10: "tb_12_1",
            20: "tb_12_2",
            50: "tb_12_3",
        },
        Framing.FP: {
            0.1: "ta_12_1",
            0.5: "ta_12_2",
            0.9: "ta_12_3",
        },
    },
    Question.STARCRAFT: {
        Framing.FY: {
            10: "tb_13_1",
            20: "tb_13_2",
            50: "tb_13_3",
        },
        Framing.FP: {
            0.1: "ta_13_1",
            0.5: "ta_13_2",
            0.9: "ta_13_3",
        },
    },
    Question.GENERAL_GAME_PLAYING: {
        Framing.FY: {
            10: "tb_14_1",
            20: "tb_14_2",
            50: "tb_14_3",
        },
        Framing.FP: {
            0.1: "ta_14_1",
            0.5: "ta_14_2",
            0.9: "ta_14_3",
        },
    },
    Question.ANGRY_BIRDS: {
        Framing.FY: {
            10: "tb_15_1",
            20: "tb_15_2",
            50: "tb_15_3",
        },
        Framing.FP: {
            0.1: "ta_15_1",
            0.5: "ta_15_2",
            0.9: "ta_15_3",
        },
    },
    Question.ATARI_GAMES: {
        Framing.FY: {
            10: "tb_16_1",
            20: "tb_16_2",
            50: "tb_16_3",
        },
        Framing.FP: {
            0.1: "ta_16_1",
            0.5: "ta_16_2",
            0.9: "ta_16_3",
        },
    },
    Question.ATARI_GAMES_NOVICE: {
        Framing.FY: {
            10: "tb_17_1",
            20: "tb_17_2",
            50: "tb_17_3",
        },
        Framing.FP: {
            0.1: "ta_17_1",
            0.5: "ta_17_2",
            0.9: "ta_17_3",
        },
    },
    Question.LAUNDRY_FOLDING: {
        Framing.FY: {
            10: "tb_18_1",
            20: "tb_18_2",
            50: "tb_18_3",
        },
        Framing.FP: {
            0.1: "ta_18_1",
            0.5: "ta_18_2",
            0.9: "ta_18_3",
        },
    },
    Question.BIPEDAL_RUNNING: {
        Framing.FY: {
            10: "tb_19_1",
            20: "tb_19_2",
            50: "tb_19_3",
        },
        Framing.FP: {
            0.1: "ta_19_1",
            0.5: "ta_19_2",
            0.9: "ta_19_3",
        },
    },
    Question.LEGO_ASSEMBLY: {
        Framing.FY: {
            10: "tb_20_1",
            20: "tb_20_2",
            50: "tb_20_3",
        },
        Framing.FP: {
            0.1: "ta_20_1",
            0.5: "ta_20_2",
            0.9: "ta_20_3",
        },
    },
    Question.SORTING_ALGORITHM: {
        Framing.FY: {
            10: "tb_21_1",
            20: "tb_21_2",
            50: "tb_21_3",
        },
        Framing.FP: {
            0.1: "ta_21_1",
            0.5: "ta_21_2",
            0.9: "ta_21_3",
        },
    },
    Question.CODE_GENERATION: {
        Framing.FY: {
            10: "tb_22_1",
            20: "tb_22_2",
            50: "tb_22_3",
        },
        Framing.FP: {
            0.1: "ta_22_1",
            0.5: "ta_22_2",
            0.9: "ta_22_3",
        },
    },
    Question.FACTOID_QA: {
        Framing.FY: {
            10: "tb_23_1",
            20: "tb_23_2",
            50: "tb_23_3",
        },
        Framing.FP: {
            0.1: "ta_23_1",
            0.5: "ta_23_2",
            0.9: "ta_23_3",
        },
    },
    Question.OPEN_ENDED_QA: {
        Framing.FY: {
            10: "tb_24_1",
            20: "tb_24_2",
            50: "tb_24_3",
        },
        Framing.FP: {
            0.1: "ta_24_1",
            0.5: "ta_24_2",
            0.9: "ta_24_3",
        },
    },
    Question.AMBIGUOUS_QA: {
        Framing.FY: {
            10: "tb_25_1",
            20: "tb_25_2",
            50: "tb_25_3",
        },
        Framing.FP: {
            0.1: "ta_25_1",
            0.5: "ta_25_2",
            0.9: "ta_25_3",
        },
    },
    Question.HIGH_SCHOOL_ESSAY: {
        Framing.FY: {
            10: "tb_26_1",
            20: "tb_26_2",
            50: "tb_26_3",
        },
        Framing.FP: {
            0.1: "ta_26_1",
            0.5: "ta_26_2",
            0.9: "ta_26_3",
        },
    },
    Question.POP_SONG_COMPOSITION: {
        Framing.FY: {
            10: "tb_27_1",
            20: "tb_27_2",
            50: "tb_27_3",
        },
        Framing.FP: {
            0.1: "ta_27_1",
            0.5: "ta_27_2",
            0.9: "ta_27_3",
        },
    },
    Question.ARTIST_SPECIFIC_SONG: {
        Framing.FY: {
            10: "tb_28_1",
            20: "tb_28_2",
            50: "tb_28_3",
        },
        Framing.FP: {
            0.1: "ta_28_1",
            0.5: "ta_28_2",
            0.9: "ta_28_3",
        },
    },
    Question.BESTSELLER_NOVEL: {
        Framing.FY: {
            10: "tb_29_1",
            20: "tb_29_2",
            50: "tb_29_3",
        },
        Framing.FP: {
            0.1: "ta_29_1",
            0.5: "ta_29_2",
            0.9: "ta_29_3",
        },
    },
    Question.GAME_STRATEGY_EXPLANATION: {
        Framing.FY: {
            10: "tb_30_1",
            20: "tb_30_2",
            50: "tb_30_3",
        },
        Framing.FP: {
            0.1: "ta_30_1",
            0.5: "ta_30_2",
            0.9: "ta_30_3",
        },
    },
    Question.POKER: {
        Framing.FY: {
            10: "tb_31_1",
            20: "tb_31_2",
            50: "tb_31_3",
        },
        Framing.FP: {
            0.1: "ta_31_1",
            0.5: "ta_31_2",
            0.9: "ta_31_3",
        },
    },
    Question.PHYSICS_EQUATIONS_DISCOVERY: {
        Framing.FY: {
            10: "tb_32_1",
            20: "tb_32_2",
            50: "tb_32_3",
        },
        Framing.FP: {
            0.1: "ta_32_1",
            0.5: "ta_32_2",
            0.9: "ta_32_3",
        },
    },
    Question.ELECTRICAL_WIRING: {
        Framing.FY: {
            10: "tb_35_1",
            20: "tb_35_2",
            50: "tb_35_3",
        },
        Framing.FP: {
            0.1: "ta_35_1",
            0.5: "ta_35_2",
            0.9: "ta_35_3",
        },
    },
    Question.LLM_FINE_TUNING: {
        Framing.FY: {
            10: "tb_37_1",
            20: "tb_37_2",
            50: "tb_37_3",
        },
        Framing.FP: {
            0.1: "ta_37_1",
            0.5: "ta_37_2",
            0.9: "ta_37_3",
        },
    },
    Question.MATH_PROBLEM_SOLVING: {
        Framing.FY: {
            10: "tb_34_1",
            20: "tb_34_2",
            50: "tb_34_3",
        },
        Framing.FP: {
            0.1: "ta_34_1",
            0.5: "ta_34_2",
            0.9: "ta_34_3",
        },
    },
    Question.ML_STUDY_REPLICATION: {
        Framing.FY: {
            10: "tb_36_1",
            20: "tb_36_2",
            50: "tb_36_3",
        },
        Framing.FP: {
            0.1: "ta_36_1",
            0.5: "ta_36_2",
            0.9: "ta_36_3",
        },
    },
    Question.WEBSITE_CREATION: {
        Framing.FY: {
            10: "tb_38_1",
            20: "tb_38_2",
            50: "tb_38_3",
        },
        Framing.FP: {
            0.1: "ta_38_1",
            0.5: "ta_38_2",
            0.9: "ta_38_3",
        },
    },
    Question.SECURITY_FLAW_DETECTION: {
        Framing.FY: {
            10: "tb_39_1",
            20: "tb_39_2",
            50: "tb_39_3",
        },
        Framing.FP: {
            0.1: "ta_39_1",
            0.5: "ta_39_2",
            0.9: "ta_39_3",
        },
    },
}
