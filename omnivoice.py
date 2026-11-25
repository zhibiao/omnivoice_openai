#!/usr/bin/env python3
"""Standalone OmniVoice inference implementation.

This file contains the complete inference path from the native OmniVoice
implementation: model loading, voice-clone prompts, text normalization,
duration estimation, iterative masked-token generation, audio tokenization,
decoding, and output post-processing. It intentionally does not import the
``omnivoice`` package.

Loaded as a module, it exposes ``OmniVoice``, ``OmniVoiceConfig``,
``OmniVoiceGenerationConfig``, and ``VoiceClonePrompt``. Command-line
inference is provided separately by ``inference.py``.
"""

from __future__ import annotations

import bisect
import difflib
import io
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, fields
from functools import lru_cache, partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from pydub.silence import (
    detect_leading_silence,
    detect_nonsilent,
    split_on_silence,
)
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

try:
    from torch.nn.attention.flex_attention import create_block_mask

    _flex_attention_available = True
except ImportError:
    _flex_attention_available = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language map
# ---------------------------------------------------------------------------

LANG_NAME_TO_ID = {
    "abadi": "kbt",
    "abkhazian": "ab",
    "abron": "abr",
    "abua": "abn",
    "adamawa fulfulde": "fub",
    "adyghe": "ady",
    "afade": "aal",
    "afrikaans": "af",
    "agwagwune": "yay",
    "aja (benin)": "ajg",
    "akebu": "keu",
    "alago": "ala",
    "albanian": "sq",
    "algerian arabic": "arq",
    "algerian saharan arabic": "aao",
    "ambo-pasco quechua": "qva",
    "ambonese malay": "abs",
    "amdo tibetan": "adx",
    "amharic": "am",
    "anaang": "anw",
    "angika": "anp",
    "antankarana malagasy": "xmv",
    "aragonese": "an",
    "arbëreshë albanian": "aae",
    "arequipa-la unión quechua": "qxu",
    "armenian": "hy",
    "ashe": "ahs",
    "ashéninka perené": "prq",
    "askopan": "eiv",
    "assamese": "as",
    "asturian": "ast",
    "atayal": "tay",
    "awak": "awo",
    "ayacucho quechua": "quy",
    "azerbaijani": "az",
    "baatonum": "bba",
    "bacama": "bcy",
    "bade": "bde",
    "bafia": "ksf",
    "bafut": "bfd",
    "bagirmi fulfulde": "fui",
    "bago-kusuntu": "bqg",
    "baharna arabic": "abv",
    "bakoko": "bkh",
    "balanta-ganja": "bjt",
    "balti": "bft",
    "bamenyam": "bce",
    "bamun": "bax",
    "bangwinji": "bsj",
    "banjar": "bjn",
    "bankon": "abb",
    "baoulé": "bci",
    "bara malagasy": "bhr",
    "barok": "bjk",
    "basa (cameroon)": "bas",
    "basa (nigeria)": "bzw",
    "bashkir": "ba",
    "basque": "eu",
    "batak mandailing": "btm",
    "batanga": "bnm",
    "bateri": "btv",
    "bats": "bbl",
    "bayot": "bda",
    "bebele": "beb",
    "belarusian": "be",
    "bengali": "bn",
    "betawi": "bew",
    "bhili": "bhb",
    "bhojpuri": "bho",
    "bilur": "bxf",
    "bima": "bhp",
    "bodo": "brx",
    "boghom": "bux",
    "bokyi": "bky",
    "bomu": "bmq",
    "bondei": "bou",
    "borgu fulfulde": "fue",
    "bosnian": "bs",
    "brahui": "brh",
    "braj": "bra",
    "breton": "br",
    "buduma": "bdm",
    "buginese": "bug",
    "bukharic": "bhh",
    "bulgarian": "bg",
    "bulu (cameroon)": "bum",
    "bundeli": "bns",
    "bunun": "bnn",
    "bura-pabir": "bwr",
    "burak": "bys",
    "burmese": "my",
    "burushaski": "bsk",
    "cacaloxtepec mixtec": "miu",
    "cajatambo north lima quechua": "qvl",
    "cakfem-mushere": "cky",
    "cameroon pidgin": "wes",
    "campidanese sardinian": "sro",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "cen": "cen",
    "central kurdish": "ckb",
    "central nahuatl": "nhn",
    "central pame": "pbs",
    "central pashto": "pst",
    "central puebla nahuatl": "ncx",
    "central tarahumara": "tar",
    "central yupik": "esu",
    "central-eastern niger fulfulde": "fuq",
    "chadian arabic": "shu",
    "chichewa": "ny",
    "chichicapan zapotec": "zpv",
    "chiga": "cgg",
    "chimalapa zoque": "zoh",
    "chimborazo highland quichua": "qug",
    "chinese": "zh",
    "chiquián ancash quechua": "qxa",
    "chitwania tharu": "the",
    "chokwe": "cjk",
    "chuvash": "cv",
    "cibak": "ckl",
    "coastal konjo": "kjc",
    "copainalá zoque": "zoc",
    "cornish": "kw",
    "corongo ancash quechua": "qwa",
    "croatian": "hr",
    "cross river mbembe": "mfn",
    "cuyamecalco mixtec": "xtu",
    "czech": "cs",
    "dadiya": "dbd",
    "dagbani": "dag",
    "dameli": "dml",
    "danish": "da",
    "dargwa": "dar",
    "dazaga": "dzg",
    "deccan": "dcc",
    "degema": "deg",
    "dera (nigeria)": "kna",
    "dghwede": "dgh",
    "dhatki": "mki",
    "dhivehi": "dv",
    "dhofari arabic": "adf",
    "dijim-bwilim": "cfa",
    "dogri": "dgo",
    "domaaki": "dmk",
    "dotyali": "dty",
    "duala": "dua",
    "dutch": "nl",
    "dũya": "ldb",
    "dyula": "dyu",
    "eastern balochi": "bgp",
    "eastern bolivian guaraní": "gui",
    "eastern egyptian bedawi arabic": "avl",
    "eastern krahn": "kqo",
    "eastern mari": "mhr",
    "eastern yiddish": "ydd",
    "ebrié": "ebr",
    "eggon": "ego",
    "egyptian arabic": "arz",
    "ejagham": "etu",
    "eleme": "elm",
    "eloyi": "afo",
    "embu": "ebu",
    "english": "en",
    "erzya": "myv",
    "esan": "ish",
    "esperanto": "eo",
    "estonian": "et",
    "eton (cameroon)": "eto",
    "ewondo": "ewo",
    "extremaduran": "ext",
    "fang (equatorial guinea)": "fan",
    "fanti": "fat",
    "farefare": "gur",
    "fe'fe'": "fmp",
    "filipino": "fil",
    "filomena mata-coahuitlán totonac": "tlp",
    "finnish": "fi",
    "fipa": "fip",
    "french": "fr",
    "fulah": "ff",
    "galician": "gl",
    "gambian wolof": "wof",
    "ganda": "lg",
    "garhwali": "gbm",
    "gawar-bati": "gwt",
    "gawri": "gwc",
    "gbagyi": "gbr",
    "gbari": "gby",
    "geji": "gyz",
    "gen": "gej",
    "georgian": "ka",
    "german": "de",
    "geser-gorom": "ges",
    "gheg albanian": "aln",
    "ghomálá'": "bbj",
    "gidar": "gid",
    "glavda": "glw",
    "goan konkani": "gom",
    "goaria": "gig",
    "goemai": "ank",
    "gola": "gol",
    "greek": "el",
    "guarani": "gn",
    "guduf-gava": "gdf",
    "guerrero amuzgo": "amu",
    "gujarati": "gu",
    "gujari": "gju",
    "gulf arabic": "afb",
    "gurgula": "ggg",
    "gusii": "guz",
    "gusilay": "gsl",
    "gweno": "gwe",
    "güilá zapotec": "ztu",
    "hadothi": "hoj",
    "hahon": "hah",
    "haitian": "ht",
    "hakha chin": "cnh",
    "hakö": "hao",
    "halia": "hla",
    "hausa": "ha",
    "hawaiian": "haw",
    "hazaragi": "haz",
    "hebrew": "he",
    "hemba": "hem",
    "herero": "hz",
    "highland konjo": "kjk",
    "hijazi arabic": "acw",
    "hindi": "hi",
    "huarijio": "var",
    "huautla mazatec": "mau",
    "huaxcaleca nahuatl": "nhq",
    "huba": "hbb",
    "huitepec mixtec": "mxs",
    "hula": "hul",
    "hungarian": "hu",
    "hunjara-kaina ke": "hkk",
    "hwana": "hwo",
    "ibibio": "ibb",
    "icelandic": "is",
    "idakho-isukha-tiriki": "ida",
    "idoma": "idu",
    "igbo": "ig",
    "igo": "ahl",
    "ikposo": "kpo",
    "ikwere": "ikw",
    "imbabura highland quichua": "qvi",
    "indonesian": "id",
    "indus kohistani": "mvy",
    "interlingua (international auxiliary language association)": "ia",
    "inupiaq": "ik",
    "irish": "ga",
    "iron ossetic": "os",
    "isekiri": "its",
    "isoko": "iso",
    "italian": "it",
    "ito": "itw",
    "itzá": "itz",
    "ixtayutla mixtec": "vmj",
    "izon": "ijc",
    "jambi malay": "jax",
    "japanese": "ja",
    "jaqaru": "jqr",
    "jauja wanca quechua": "qxw",
    "jaunsari": "jns",
    "javanese": "jv",
    "jiba": "juo",
    "jju": "kaj",
    "judeo-moroccan arabic": "aju",
    "juxtlahuaca mixtec": "vmc",
    "kabardian": "kbd",
    "kabras": "lkb",
    "kabuverdianu": "kea",
    "kabyle": "kab",
    "kachi koli": "gjk",
    "kairak": "ckr",
    "kalabari": "ijn",
    "kalasha": "kls",
    "kalenjin": "kln",
    "kalkoti": "xka",
    "kamba": "kam",
    "kamo": "kcq",
    "kanauji": "bjj",
    "kanembu": "kbl",
    "kannada": "kn",
    "karekare": "kai",
    "kashmiri": "ks",
    "kathoriya tharu": "tkt",
    "kati": "bsh",
    "kazakh": "kk",
    "keiyo": "eyo",
    "khams tibetan": "khg",
    "khana": "ogo",
    "khetrani": "xhe",
    "khmer": "km",
    "khowar": "khw",
    "kinga": "zga",
    "kinnauri": "kfk",
    "kinyarwanda": "rw",
    "kirghiz": "ky",
    "kirya-konzəl": "fkk",
    "kochila tharu": "thq",
    "kohistani shina": "plk",
    "kohumono": "bcs",
    "kok borok": "trp",
    "kol (papua new guinea)": "kol",
    "kom (cameroon)": "bkm",
    "koma": "kmy",
    "konkani": "knn",
    "konzo": "koo",
    "korean": "ko",
    "korwa": "kfp",
    "kota (india)": "kfe",
    "koti": "eko",
    "kuanua": "ksd",
    "kuanyama": "kj",
    "kui (india)": "uki",
    "kulung (nigeria)": "bbu",
    "kuot": "kto",
    "kushi": "kuh",
    "kwambi": "kwm",
    "kwasio": "nmg",
    "lala-roba": "lla",
    "lamang": "hia",
    "lao": "lo",
    "larike-wakasihu": "alo",
    "lasi": "lss",
    "latgalian": "ltg",
    "latvian": "lv",
    "levantine arabic": "apc",
    "liana-seti": "ste",
    "liberia kpelle": "xpe",
    "liberian english": "lir",
    "libyan arabic": "ayl",
    "ligurian": "lij",
    "lijili": "mgi",
    "lingala": "ln",
    "lithuanian": "lt",
    "loarki": "lrk",
    "logooli": "rag",
    "logudorese sardinian": "src",
    "loja highland quichua": "qvj",
    "loloda": "loa",
    "longuda": "lnu",
    "loxicha zapotec": "ztp",
    "luba-lulua": "lua",
    "luo": "luo",
    "lushai": "lus",
    "luxembourgish": "lb",
    "maasina fulfulde": "ffm",
    "maba (chad)": "mde",
    "macedo-romanian": "rup",
    "macedonian": "mk",
    "mada (cameroon)": "mxu",
    "mafa": "maf",
    "maithili": "mai",
    "malay": "ms",
    "malayalam": "ml",
    "mali": "gcc",
    "malinaltepec me'phaa": "tcf",
    "maltese": "mt",
    "mandara": "tbf",
    "mandjak": "mfv",
    "manggarai": "mqy",
    "manipuri": "mni",
    "mansoanka": "msw",
    "manx": "gv",
    "maori": "mi",
    "marathi": "mr",
    "marghi central": "mrt",
    "marghi south": "mfm",
    "maria (india)": "mrr",
    "marwari (pakistan)": "mve",
    "masana": "mcn",
    "masikoro malagasy": "msh",
    "matsés": "mcf",
    "mazaltepec zapotec": "zpy",
    "mazatlán mazatec": "vmz",
    "mazatlán mixe": "mzl",
    "mbe": "mfo",
    "mbo (cameroon)": "mbo",
    "mbum": "mdd",
    "medumba": "byv",
    "mekeo": "mek",
    "meru": "mer",
    "mesopotamian arabic": "acm",
    "mewari": "mtr",
    "min nan chinese": "nan",
    "mingrelian": "xmf",
    "mitlatongo mixtec": "vmm",
    "miya": "mkf",
    "mokpwe": "bri",
    "moksha": "mdf",
    "mom jango": "ver",
    "mongolian": "mn",
    "moroccan arabic": "ary",
    "motu": "meu",
    "mpiemo": "mcx",
    "mpumpong": "mgg",
    "mundang": "mua",
    "mungaka": "mhk",
    "musey": "mse",
    "musgu": "mug",
    "musi": "mui",
    "naba": "mne",
    "najdi arabic": "ars",
    "nalik": "nal",
    "nawdm": "nmz",
    "ndonga": "ng",
    "neapolitan": "nap",
    "nepali": "npi",
    "ngamo": "nbh",
    "ngas": "anc",
    "ngiemboon": "nnh",
    "ngizim": "ngi",
    "ngomba": "jgo",
    "ngombale": "nla",
    "nigerian fulfulde": "fuv",
    "nigerian pidgin": "pcm",
    "nimadi": "noe",
    "nobiin": "fia",
    "north mesopotamian arabic": "ayp",
    "north moluccan malay": "max",
    "northern betsimisaraka malagasy": "bmm",
    "northern hindko": "hno",
    "northern kurdish": "kmr",
    "northern pame": "pmq",
    "northern pashto": "pbu",
    "northern uzbek": "uzn",
    "northwest gbaya": "gya",
    "norwegian": "no",
    "norwegian bokmål": "nb",
    "norwegian nynorsk": "nn",
    "notsi": "ncf",
    "nyankpa": "yes",
    "nyungwe": "nyu",
    "nzanyi": "nja",
    "nüpode huitoto": "hux",
    "occitan": "oc",
    "od": "odk",
    "odia": "ory",
    "odual": "odu",
    "omani arabic": "acx",
    "orizaba nahuatl": "nlv",
    "orma": "orc",
    "ormuri": "oru",
    "oromo": "om",
    "pahari-potwari": "phr",
    "paiwan": "pwn",
    "panjabi": "pa",
    "papuan malay": "pmy",
    "parkari koli": "kvx",
    "pedi": "nso",
    "pero": "pip",
    "persian": "fa",
    "petats": "pex",
    "phalura": "phl",
    "piemontese": "pms",
    "piya-kwonci": "piy",
    "plateau malagasy": "plt",
    "polish": "pl",
    "poqomam": "poc",
    "portuguese": "pt",
    "pulaar": "fuc",
    "pular": "fuf",
    "puno quechua": "qxp",
    "pushto": "ps",
    "pökoot": "pko",
    "qaqet": "byx",
    "quiotepec chinantec": "chq",
    "rana tharu": "thr",
    "rangi": "lag",
    "rapoisi": "kyx",
    "ratahan": "rth",
    "rayón zoque": "zor",
    "romanian": "ro",
    "romansh": "rm",
    "rombo": "rof",
    "rotokas": "roo",
    "rukai": "dru",
    "russian": "ru",
    "sacapulteco": "quv",
    "saidi arabic": "aec",
    "sakalava malagasy": "skg",
    "sakizaya": "szy",
    "saleman": "sau",
    "samba daka": "ccg",
    "samba leko": "ndi",
    "san felipe otlaltepec popoloca": "pow",
    "san francisco del mar huave": "hue",
    "san juan atzingo popoloca": "poe",
    "san martín itunyoso triqui": "trq",
    "san miguel el grande mixtec": "mig",
    "sansi": "ssi",
    "sanskrit": "sa",
    "santa ana de tusi pasco quechua": "qxt",
    "santa catarina albarradas zapotec": "ztn",
    "santali": "sat",
    "santiago del estero quichua": "qus",
    "saposa": "sps",
    "saraiki": "skr",
    "sardinian": "sc",
    "saya": "say",
    "sediq": "trv",
    "serbian": "sr",
    "seri": "sei",
    "shina": "scl",
    "shona": "sn",
    "siar-lak": "sjr",
    "sibe": "nco",
    "sicilian": "scn",
    "sihuas ancash quechua": "qws",
    "sikkimese": "sip",
    "sinaugoro": "snc",
    "sindhi": "sd",
    "sindhi bhil": "sbn",
    "sinhala": "si",
    "sinicahua mixtec": "xti",
    "sipacapense": "qum",
    "siwai": "siw",
    "slovak": "sk",
    "slovenian": "sl",
    "solos": "sol",
    "somali": "so",
    "soninke": "snk",
    "south giziga": "giz",
    "south ucayali ashéninka": "cpy",
    "southeastern nochixtlán mixtec": "mxy",
    "southern betsimisaraka malagasy": "bzc",
    "southern pashto": "pbt",
    "southern pastaza quechua": "qup",
    "soyaltepec mazatec": "vmp",
    "spanish": "es",
    "standard arabic": "arb",
    "standard moroccan tamazight": "zgh",
    "sudanese arabic": "apd",
    "sulka": "sua",
    "svan": "sva",
    "swahili": "sw",
    "swedish": "sv",
    "tae'": "rob",
    "tahaggart tamahaq": "thv",
    "taita": "dav",
    "tajik": "tg",
    "tamil": "ta",
    "tandroy-mahafaly malagasy": "tdx",
    "tangale": "tan",
    "tanosy malagasy": "txy",
    "tarok": "yer",
    "tatar": "tt",
    "tedaga": "tuq",
    "telugu": "te",
    "tem": "kdh",
    "teop": "tio",
    "tepeuxila cuicatec": "cux",
    "tepinapa chinantec": "cte",
    "tera": "ttr",
    "terei": "buo",
    "termanu": "twu",
    "tesaka malagasy": "tkg",
    "tetelcingo nahuatl": "nhg",
    "teutila cuicatec": "cut",
    "thai": "th",
    "tibetan": "bo",
    "tidaá mixtec": "mtx",
    "tidore": "tvo",
    "tigak": "tgc",
    "tigre": "tig",
    "tigrinya": "ti",
    "tilquiapan zapotec": "zts",
    "tinputz": "tpz",
    "tlacoapa me'phaa": "tpl",
    "tlacoatzintepec chinantec": "ctl",
    "tlingit": "tli",
    "toki pona": "tok",
    "tomoip": "tqp",
    "tondano": "tdn",
    "tonsea": "txs",
    "tooro": "ttj",
    "torau": "ttu",
    "torwali": "trw",
    "tsimihety malagasy": "xmw",
    "tsotso": "lto",
    "tswana": "tn",
    "tugen": "tuy",
    "tuki": "bag",
    "tula": "tul",
    "tulu": "tcy",
    "tunen": "tvu",
    "tungag": "lcm",
    "tunisian arabic": "aeb",
    "tupuri": "tui",
    "turkana": "tuv",
    "turkish": "tr",
    "turkmen": "tk",
    "tututepec mixtec": "mtu",
    "twi": "tw",
    "ubaghara": "byc",
    "uighur": "ug",
    "ukrainian": "uk",
    "umbundu": "umb",
    "upper sorbian": "hsb",
    "urdu": "ur",
    "ushojo": "ush",
    "uzbek": "uz",
    "vai": "vai",
    "vietnamese": "vi",
    "votic": "vot",
    "võro": "vro",
    "waci gbe": "wci",
    "wadiyara koli": "kxp",
    "waja": "wja",
    "wakhi": "wbl",
    "wanga": "lwg",
    "wapan": "juk",
    "warji": "wji",
    "welsh": "cy",
    "wemale": "weo",
    "western frisian": "fy",
    "western highland purepecha": "pua",
    "western juxtlahuaca mixtec": "jmx",
    "western maninkakan": "mlq",
    "western mari": "mrj",
    "western niger fulfulde": "fuh",
    "western panjabi": "pnb",
    "wolof": "wo",
    "wuzlam": "udl",
    "xanaguía zapotec": "ztg",
    "xhosa": "xh",
    "yace": "ekr",
    "yakut": "sah",
    "yalahatan": "jal",
    "yanahuanca pasco quechua": "qur",
    "yangben": "yav",
    "yaqui": "yaq",
    "yauyos quechua": "qux",
    "yekhee": "ets",
    "yiddish": "yi",
    "yidgha": "ydg",
    "yoruba": "yo",
    "yutanduchi mixtec": "mab",
    "zacatlán-ahuacatlán-tepetzintla nahuatl": "nhi",
    "zarma": "dje",
    "zaza": "zza",
    "zulu": "zu",
    "ömie": "aom",
}

LANG_NAMES = set(LANG_NAME_TO_ID.keys())
LANG_IDS = set(LANG_NAME_TO_ID.values())

# Exceptions where .title() doesn't match the canonical casing from the TSV.
_TITLE_EXCEPTIONS = {
    "fe'fe'": "Fe'fe'",
    "dũya": "Dũya",
    "santiago del estero quichua": "Santiago del Estero Quichua",
    "santa ana de tusi pasco quechua": "Santa Ana de Tusi Pasco Quechua",
    "malinaltepec me'phaa": "Malinaltepec Me'phaa",
    "tlacoapa me'phaa": "Tlacoapa Me'phaa",
}


def lang_display_name(name: str) -> str:
    """Return a display-friendly version of a lowercase language name.

    Uses .title() for most names, with manual exceptions for cases like
    apostrophes and small words (de, del) that should stay lowercase.
    """
    return _TITLE_EXCEPTIONS.get(name, name.title())


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------

class RuleDurationEstimator:
    def __init__(self):
        # ==========================================
        # 1. Phonetic Weights Table
        # ==========================================
        # The weight represents the relative speaking time compared to
        # a standard Latin letter.
        # Benchmark: 1.0 = One Latin Character (~40-50ms)
        self.weights = {
            # --- Logographic (1 char = full syllable/word) ---
            "cjk": 3.0,  # Chinese, Japanese Kanji, etc.
            # --- Syllabic / Blocks
            "hangul": 2.5,  # Korean Hangul
            "kana": 2.2,  # Japanese Hiragana/Katakana
            "ethiopic": 3.0,  # Amharic/Ge'ez
            "yi": 3.0,  # Yi script
            # --- Abugida (Consonant-Vowel complexes) ---
            "indic": 1.8,  # Hindi, Bengali, Tamil, etc.
            "thai_lao": 1.5,  # Thai, Lao
            "khmer_myanmar": 1.8,  # Khmer, Myanmar
            # --- Abjad (Consonant-heavy) ---
            "arabic": 1.5,  # Arabic, Persian, Urdu
            "hebrew": 1.5,  # Hebrew
            # --- Alphabet (Segmental) ---
            "latin": 1.0,  # English, Spanish, French, Vietnamese, etc. (Baseline)
            "cyrillic": 1.0,  # Russian, Ukrainian
            "greek": 1.0,  # Greek
            "armenian": 1.0,  # Armenian
            "georgian": 1.0,  # Georgian
            # --- Symbols & Misc ---
            "punctuation": 0.5,  # Pause capability
            "space": 0.2,  # Word boundary/Breath (0.05 / 0.22)
            "digit": 3.5,  # Numbers
            "mark": 0.0,  # Diacritics/Accents (Silent modifiers)
            "default": 1.0,  # Fallback for unknown scripts
        }

        # ==========================================
        # 2. Unicode Range Mapping
        # ==========================================
        # Format: (End_Codepoint, Type_Key)
        # Used for fast binary search (bisect).
        self.ranges = [
            (0x02AF, "latin"),  # Latin (Basic, Supplement, Ext, IPA)
            (0x03FF, "greek"),  # Greek & Coptic
            (0x052F, "cyrillic"),  # Cyrillic
            (0x058F, "armenian"),  # Armenian
            (0x05FF, "hebrew"),  # Hebrew
            (0x077F, "arabic"),  # Arabic, Syriac, Arabic Supplement
            (0x089F, "arabic"),  # Arabic Extended-B (+ Syriac Supp)
            (0x08FF, "arabic"),  # Arabic Extended-A
            (0x097F, "indic"),  # Devanagari
            (0x09FF, "indic"),  # Bengali
            (0x0A7F, "indic"),  # Gurmukhi
            (0x0AFF, "indic"),  # Gujarati
            (0x0B7F, "indic"),  # Oriya
            (0x0BFF, "indic"),  # Tamil
            (0x0C7F, "indic"),  # Telugu
            (0x0CFF, "indic"),  # Kannada
            (0x0D7F, "indic"),  # Malayalam
            (0x0DFF, "indic"),  # Sinhala
            (0x0EFF, "thai_lao"),  # Thai & Lao
            (0x0FFF, "indic"),  # Tibetan (Abugida)
            (0x109F, "khmer_myanmar"),  # Myanmar
            (0x10FF, "georgian"),  # Georgian
            (0x11FF, "hangul"),  # Hangul Jamo
            (0x137F, "ethiopic"),  # Ethiopic
            (0x139F, "ethiopic"),  # Ethiopic Supplement
            (0x13FF, "default"),  # Cherokee
            (0x167F, "default"),  # Canadian Aboriginal Syllabics
            (0x169F, "default"),  # Ogham
            (0x16FF, "default"),  # Runic
            (0x171F, "default"),  # Tagalog (Baybayin)
            (0x173F, "default"),  # Hanunoo
            (0x175F, "default"),  # Buhid
            (0x177F, "default"),  # Tagbanwa
            (0x17FF, "khmer_myanmar"),  # Khmer
            (0x18AF, "default"),  # Mongolian
            (0x18FF, "default"),  # Canadian Aboriginal Syllabics Ext
            (0x194F, "indic"),  # Limbu
            (0x19DF, "indic"),  # Tai Le & New Tai Lue
            (0x19FF, "khmer_myanmar"),  # Khmer Symbols
            (0x1A1F, "indic"),  # Buginese
            (0x1AAF, "indic"),  # Tai Tham
            (0x1B7F, "indic"),  # Balinese
            (0x1BBF, "indic"),  # Sundanese
            (0x1BFF, "indic"),  # Batak
            (0x1C4F, "indic"),  # Lepcha
            (0x1C7F, "indic"),  # Ol Chiki (Santali)
            (0x1C8F, "cyrillic"),  # Cyrillic Extended-C
            (0x1CBF, "georgian"),  # Georgian Extended
            (0x1CCF, "indic"),  # Sundanese Supplement
            (0x1CFF, "indic"),  # Vedic Extensions
            (0x1D7F, "latin"),  # Phonetic Extensions
            (0x1DBF, "latin"),  # Phonetic Extensions Supplement
            (0x1DFF, "default"),  # Combining Diacritical Marks Supplement
            (0x1EFF, "latin"),  # Latin Extended Additional (Vietnamese)
            (0x309F, "kana"),  # Hiragana
            (0x30FF, "kana"),  # Katakana
            (0x312F, "cjk"),  # Bopomofo (Pinyin)
            (0x318F, "hangul"),  # Hangul Compatibility Jamo
            (0x9FFF, "cjk"),  # CJK Unified Ideographs (Main)
            (0xA4CF, "yi"),  # Yi Syllables
            (0xA4FF, "default"),  # Lisu
            (0xA63F, "default"),  # Vai
            (0xA69F, "cyrillic"),  # Cyrillic Extended-B
            (0xA6FF, "default"),  # Bamum
            (0xA7FF, "latin"),  # Latin Extended-D
            (0xA82F, "indic"),  # Syloti Nagri
            (0xA87F, "default"),  # Phags-pa
            (0xA8DF, "indic"),  # Saurashtra
            (0xA8FF, "indic"),  # Devanagari Extended
            (0xA92F, "indic"),  # Kayah Li
            (0xA95F, "indic"),  # Rejang
            (0xA97F, "hangul"),  # Hangul Jamo Extended-A
            (0xA9DF, "indic"),  # Javanese
            (0xA9FF, "khmer_myanmar"),  # Myanmar Extended-B
            (0xAA5F, "indic"),  # Cham
            (0xAA7F, "khmer_myanmar"),  # Myanmar Extended-A
            (0xAADF, "indic"),  # Tai Viet
            (0xAAFF, "indic"),  # Meetei Mayek Extensions
            (0xAB2F, "ethiopic"),  # Ethiopic Extended-A
            (0xAB6F, "latin"),  # Latin Extended-E
            (0xABBF, "default"),  # Cherokee Supplement
            (0xABFF, "indic"),  # Meetei Mayek
            (0xD7AF, "hangul"),  # Hangul Syllables
            (0xFAFF, "cjk"),  # CJK Compatibility
            (0xFDFF, "arabic"),  # Arabic Presentation Forms-A
            (0xFE6F, "default"),  # Variation Selectors
            (0xFEFF, "arabic"),  # Arabic Presentation Forms-B
            (0xFFEF, "latin"),  # Fullwidth Latin
        ]
        self.breakpoints = [r[0] for r in self.ranges]

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char):
        """Determines the weight of a single character."""
        code = ord(char)
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]
        if code == 32:
            return self.weights["space"]

        # Ignore arabic Tatweel
        if code == 0x0640:
            return self.weights["mark"]

        category = unicodedata.category(char)

        if category.startswith("M"):
            return self.weights["mark"]

        if category.startswith("P") or category.startswith("S"):
            return self.weights["punctuation"]

        if category.startswith("Z"):
            return self.weights["space"]

        if category.startswith("N"):
            return self.weights["digit"]

        # 3. Binary search for Unicode Block (此时区间里绝不会再混进标点符号)
        idx = bisect.bisect_left(self.breakpoints, code)
        if idx < len(self.ranges):
            script_type = self.ranges[idx][1]
            return self.weights.get(script_type, self.weights["default"])

        # 4. Handle upper planes (CJK Ext B/C/D, Historic scripts)
        if code > 0x20000:
            return self.weights["cjk"]

        return self.weights["default"]

    def calculate_total_weight(self, text):
        """Sums up the normalized weights for a string."""
        return sum(self._get_char_weight(c) for c in text)

    def estimate_duration(
        self,
        target_text: str,
        ref_text: str,
        ref_duration: float,
        low_threshold: Optional[float] = 50,
        boost_strength: float = 3,
    ) -> float:
        """

        Args:
            target_text (str): The text for which we want to estimate the duration.
            ref_text (str): The reference text that was used to measure
                the ref_duration.
            ref_duration (float): The actual duration it took
                to speak the ref_text.
            low_threshold (float): The minimum duration threshold below which the
                estimation will be considered unreliable.
            boost_strength (float): Controls the power-curve boost for short durations.
                Higher values boost small durations more aggressively.
                1 = no boost (linear), 2 = sqrt-like

        Returns:
            float: The estimated duration for the target_text based
                on the ref_text and ref_duration.
        """
        if ref_duration <= 0 or not ref_text:
            return 0.0

        ref_weight = self.calculate_total_weight(ref_text)
        if ref_weight == 0:
            return 0.0

        speed_factor = ref_weight / ref_duration
        target_weight = self.calculate_total_weight(target_text)

        estimated_duration = target_weight / speed_factor
        if low_threshold is not None and estimated_duration < low_threshold:
            alpha = 1.0 / boost_strength
            return low_threshold * (estimated_duration / low_threshold) ** alpha
        else:
            return estimated_duration


# ==========================================
# Example Usage
# ==========================================


# ---------------------------------------------------------------------------
# Voice design
# ---------------------------------------------------------------------------

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

# Category = set of {english: chinese, ...} items that are mutually exclusive.
# Accent (EN-only) and dialect (ZH-only) are stored as flat sets below.
_INSTRUCT_CATEGORIES = [
    {"male": "男", "female": "女"},
    {
        "child": "儿童",
        "teenager": "少年",
        "young adult": "青年",
        "middle-aged": "中年",
        "elderly": "老年",
    },
    {
        "very low pitch": "极低音调",
        "low pitch": "低音调",
        "moderate pitch": "中音调",
        "high pitch": "高音调",
        "very high pitch": "极高音调",
    },
    {"whisper": "耳语"},
    # Accent (English-only, no Chinese counterpart)
    {
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    },
    # Dialect (Chinese-only, no English counterpart)
    {
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    },
]

_INSTRUCT_EN_TO_ZH = {}
_INSTRUCT_ZH_TO_EN = {}
_INSTRUCT_MUTUALLY_EXCLUSIVE = []
for _cat in _INSTRUCT_CATEGORIES:
    if isinstance(_cat, dict):
        _INSTRUCT_EN_TO_ZH.update(_cat)
        _INSTRUCT_ZH_TO_EN.update({v: k for k, v in _cat.items()})
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat) | set(_cat.values()))
    else:
        _INSTRUCT_MUTUALLY_EXCLUSIVE.append(set(_cat))

_INSTRUCT_ALL_VALID = (
    set(_INSTRUCT_EN_TO_ZH)
    | set(_INSTRUCT_ZH_TO_EN)
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-2]  # accents
    | _INSTRUCT_MUTUALLY_EXCLUSIVE[-1]  # dialects
)

_INSTRUCT_VALID_EN = frozenset(i for i in _INSTRUCT_ALL_VALID if not _ZH_RE.search(i))
_INSTRUCT_VALID_ZH = frozenset(i for i in _INSTRUCT_ALL_VALID if _ZH_RE.search(i))


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text processing utilities for TTS inference.

Provides:
- ``chunk_text_punctuation()``: Splits long text into model-friendly chunks at
  sentence boundaries, with abbreviation-aware punctuation splitting.
- ``add_punctuation()``: Appends missing end punctuation (Chinese or English).
- ``normalize_text()``: Optional text normalization (numbers, dates, currency,
  etc.) into their spoken form, while preserving inline control syntax.
"""

logger = logging.getLogger(__name__)


SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")

END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "“",
    "”",
    "‘",
    "’",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "……",
    "）",
    "】",
}


ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Fr.",
    "Hon.",
    "Pres.",
    "Gov.",
    "Capt.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Col.",
    "Maj.",
    "Lt.",
    "Cmdr.",
    "Sgt.",
    "Cpl.",
    "Co.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "Est.",
    "Dept.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Mt.",
    "Ft.",
    "No.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "i.e.",
    "e.g.",
    "vs.",
    "Vs.",
    "Etc.",
    "approx.",
    "fig.",
    "def.",
}


def chunk_text_punctuation(
    text: str,
    chunk_len: int,
    min_chunk_len: Optional[int] = None,
) -> List[str]:
    """
    Splits the input tokens list into chunks according to punctuations,
    avoiding splits on common abbreviations (e.g., Mr., No.).
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []

    tokens_list = list(text)

    for token in tokens_list:
        # If the first token of current sentence is punctuation,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS)
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)

            # Split the sentence in positions of punctuations.
            if token in SPLIT_PUNCTUATION:
                is_abbreviation = False

                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str:
                        last_word = temp_str.split()[-1]
                        if last_word in ABBREVIATIONS:
                            is_abbreviation = True

                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    # 4. Post-process: Check for undersized chunks and merge them
    #  with the previous chunk or next chunk (if it's the first chunk).
    if min_chunk_len is not None:
        first_chunk_short_flag = (
            len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        )
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short_flag:
                final_chunks[-1].extend(chunk)
            else:
                if len(chunk) >= min_chunk_len:
                    final_chunks.append(chunk)
                else:
                    if len(final_chunks) == 0:
                        final_chunks.append(chunk)
                    else:
                        final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    chunk_strings = [
        "".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()
    ]
    return chunk_strings


def add_punctuation(text: str):
    """Add punctuation if there is not in the end of text"""
    text = text.strip()

    if not text:
        return text

    if text[-1] not in END_PUNCTUATION:
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        text += "。" if is_chinese else "."

    return text


# ---------------------------------------------------------------------------
# Optional text normalization (opt-in via ``generate(normalize_text=True)``)
# ---------------------------------------------------------------------------
#
# Arabic numerals, dates, currency, etc. are converted into their spoken form
# so the model reads them correctly (e.g. "2345" -> "twenty three forty five",
# "199" -> the Chinese reading). Chinese/English go through WeTextProcessing;
# any other language falls back to ``num2words`` for bare integers when
# available.
#
# The OmniVoice inline control syntax must survive normalization:
#   * bracketed non-verbal tags, e.g. ``[laughter]``, ``[sigh]``;
#   * bracketed CMU pronunciation overrides, e.g. ``[B EY1 S]`` -- the stress
#     digit would otherwise be read as a number;
#   * Chinese pinyin tone markers (uppercase pinyin + tone digit) -- likewise.
# Protected spans are held out and re-inserted verbatim around normalization.

# Any ``[...]`` span covers both non-verbal tags and CMU pronunciation.
_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")
# Uppercase pinyin followed by a tone digit 1-5 (Chinese pronunciation control).
_PINYIN_TONE_RE = re.compile(r"[A-Z]+[1-5]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_TN_INSTALL_MSG = (
    "Text normalization (normalize_text=True) requires WeTextProcessing, which "
    "is not installed.\n"
    "  pip install WeTextProcessing\n"
    "WeTextProcessing depends on pynini, which has no prebuilt wheel for macOS "
    "arm64 (Apple Silicon). On macOS, install pynini from conda-forge first:\n"
    "  conda install -c conda-forge pynini\n"
    "then:  pip install WeTextProcessing"
)

# Normalizer construction builds FSTs and is comparatively slow, so instances
# are cached per language for the lifetime of the process.
_ZH_NORMALIZER = None
_EN_NORMALIZER = None


def _get_zh_normalizer():
    global _ZH_NORMALIZER
    if _ZH_NORMALIZER is None:
        try:
            from tn.chinese.normalizer import Normalizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(_TN_INSTALL_MSG) from e
        # Conservative flags: normalize numbers/symbols only. Keep interjections
        # and erhua (they are spoken), keep the user's original characters, and
        # do not delete or rewrite anything beyond numeric/symbolic tokens.
        _ZH_NORMALIZER = Normalizer(
            remove_interjections=False,
            remove_erhua=False,
            traditional_to_simple=False,
            remove_puncts=False,
            full_to_half=False,
        )
    return _ZH_NORMALIZER


def _get_en_normalizer():
    global _EN_NORMALIZER
    if _EN_NORMALIZER is None:
        try:
            from tn.english.normalizer import Normalizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(_TN_INSTALL_MSG) from e
        _EN_NORMALIZER = Normalizer()
    return _EN_NORMALIZER


def _resolve_lang_code(language: Optional[str], text: str) -> str:
    """Map a language name/code to ``"zh"``/``"en"``/other code.

    When ``language`` is ``None`` (or unrecognized), fall back to detecting
    Chinese vs. English by the presence of CJK characters.
    """
    if language is not None:
        code = language.strip().lower()
        if code and code != "none":
            if code in ("zh", "en"):
                return code
            try:
                pass

                if code in LANG_IDS:
                    return code
                if code in LANG_NAME_TO_ID:
                    return LANG_NAME_TO_ID[code]
            except Exception:  # pragma: no cover - lang_map should be importable
                pass
            return code  # assume it is already a language id, e.g. "ja", "de"
    return "zh" if _CJK_RE.search(text) else "en"


def _num2words_segment(text: str, lang: str) -> str:
    """Best-effort integer-to-words fallback for non zh/en languages."""
    try:
        from num2words import num2words
    except ImportError:
        return text  # fallback is best-effort; silently skip when unavailable

    def _repl(match):
        try:
            return num2words(int(match.group()), lang=lang)
        except Exception:
            return match.group()  # unsupported language / value: leave as-is

    return re.sub(r"\d+", _repl, text)


def _normalize_segment(fn: Callable[[str], str], segment: str) -> str:
    """Normalize one non-protected segment, never raising on bad input.

    Leading/trailing whitespace is preserved explicitly because the underlying
    normalizers strip it, which would otherwise glue words to an adjacent
    protected span (e.g. ``the [B EY1 S] guitar`` -> ``the[B EY1 S]guitar``).
    """
    if not segment.strip():
        return segment
    lead = segment[: len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()) :]
    try:
        core = fn(segment.strip())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Text normalization failed on a segment (%s); keeping it unchanged.",
            type(e).__name__,
        )
        return segment
    return lead + core + trail


def _apply_with_protection(
    text: str, fn: Callable[[str], str], protect_pinyin: bool
) -> str:
    """Run ``fn`` on ``text`` while holding out protected control spans."""
    spans = [m.span() for m in _BRACKET_TAG_RE.finditer(text)]
    if protect_pinyin:
        spans += [m.span() for m in _PINYIN_TONE_RE.finditer(text)]
    if not spans:
        return _normalize_segment(fn, text)

    # Merge overlapping/adjacent protected spans, then normalize the gaps.
    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: List[str] = []
    last = 0
    for start, end in merged:
        if start > last:
            out.append(_normalize_segment(fn, text[last:start]))
        out.append(text[start:end])  # protected span, verbatim
        last = end
    if last < len(text):
        out.append(_normalize_segment(fn, text[last:]))
    return "".join(out)


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalize numbers, dates, currency, etc. into their spoken form.

    Chinese is routed to WeTextProcessing's ``ZhNormalizer`` and English to its
    ``EnNormalizer`` (configured to only rewrite numeric/symbolic tokens). Any
    other language falls back to ``num2words`` for bare integers when it is
    installed, otherwise the text is returned unchanged.

    Inline OmniVoice control syntax is preserved: bracketed non-verbal tags
    (``[laughter]``) and CMU pronunciation overrides (``[B EY1 S]``) are passed
    through untouched, and Chinese pinyin tone markers (uppercase pinyin +
    tone digit) are protected so the tone digit is not read as a number.

    Args:
        text: Input text.
        language: Language code (``"en"``/``"zh"``) or full name (``"English"``).
            ``None`` auto-detects Chinese vs. English by script.

    Returns:
        The normalized text.

    Raises:
        ImportError: For Chinese/English when the optional WeTextProcessing
            dependency (WeTextProcessing) is not installed.
    """
    if not text or not text.strip():
        return text

    code = _resolve_lang_code(language, text)
    if code == "zh":
        normalizer = _get_zh_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=True)
    if code == "en":
        normalizer = _get_en_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=False)
    # Other languages: best-effort integer conversion via num2words.
    return _apply_with_protection(
        text, lambda s: _num2words_segment(s, code), protect_pinyin=False
    )

# Native model code refers to normalize_text through this private alias.
_normalize_text = normalize_text


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio I/O and processing utilities.

Provides functions for loading, resampling, silence removal,
chunking, cross-fading, and format conversion.

All public functions in this module operate on **numpy float32 arrays**
with shape ``(C, T)`` (channels-first).
"""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_waveform(audio_path: str):
    """Load audio from a file path, returning (data, sample_rate).

    Tries two backends in order:
    1. soundfile — covers WAV/FLAC/OGG etc., no ffmpeg needed.
    2. librosa — covers MP3/M4A etc. via audioread + ffmpeg.

    Returns:
        (data, sample_rate) where data is a numpy float32 array of
        shape (C, T).
    """
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sr  # (T, C) → (C, T)
    except Exception:
        # soundfile cannot handle MP3/M4A etc., fall back to librosa.
        import librosa

        data, sr = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data, sr


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load a waveform from file and resample to the target rate.

    Parameters:
        audio_path: path of the audio.
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    data, sr = load_waveform(audio_path)

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


def load_audio_bytes(raw: bytes, sampling_rate: int) -> np.ndarray:
    """Load audio from in-memory bytes and resample.

    Parameters:
        raw: raw audio file bytes (e.g. from WebDataset).
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    buf = io.BytesIO(raw)

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T  # (T, C) → (C, T)
    except Exception:
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


# ---------------------------------------------------------------------------
# Audio processing (all numpy in / numpy out)
# ---------------------------------------------------------------------------


def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a numpy float32 array of shape (C, T) to a pydub AudioSegment."""
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()  # interleave channels
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a numpy float32 array of shape (C, T)."""
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Remove middle silences longer than *mid_sil* ms and trim edge silences.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: middle-silence threshold in ms (0 to skip).
        lead_sil: kept leading silence in ms.
        trail_sil: kept trailing silence in ms.

    Returns:
        Numpy array with shape (C, T').
    """
    wave = numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)

    return audiosegment_to_numpy(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    """Remove edge silences, keeping *lead_sil* / *trail_sil* ms."""
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply fade-in/out and pad with silence to prevent clicks.

    Args:
        audio: numpy array of shape (C, T).
        pad_duration: silence padding duration per side (seconds).
        fade_duration: fade curve duration (seconds).
        sample_rate: audio sampling rate.

    Returns:
        Processed numpy array of shape (C, T_new).
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in

            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
        )
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio to <= *max_duration* by splitting at the largest silence gap.

    Only trims when the audio exceeds *trim_threshold* seconds.

    Args:
        audio: numpy array of shape (C, T).
        sampling_rate: audio sampling rate.
        max_duration: maximum duration in seconds.
        min_duration: minimum duration in seconds.
        trim_threshold: only trim if audio is longer than this (seconds).

    Returns:
        Trimmed numpy array.
    """
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    seg = numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    return audiosegment_to_numpy(trimmed)


def cross_fade_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    """Concatenate audio chunks with silence gaps and cross-fade at boundaries.

    Args:
        chunks: list of numpy arrays, each (C, T).
        sample_rate: audio sample rate.
        silence_duration: total silence gap duration in seconds.

    Returns:
        Merged numpy array (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]

        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
            parts[-1][..., -fout_n:] *= w_out

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
            fade_in[..., :fin_n] *= w_in

        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)

    return merged


# ---------------------------------------------------------------------------
# Native model and inference
# ---------------------------------------------------------------------------

#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core OmniVoice model implementation.

Defines the ``OmniVoice`` model class, generation config, and inference pipeline.
This is the main entry point for both inference and training:

- **Inference**: ``OmniVoice.from_pretrained()`` loads the model, then
  ``model.generate()`` supports voice cloning, voice design, and auto voice.
- **Training**: ``model.forward()`` computes the training loss; the model is
  built and used by ``omnivoice.training.builder`` and ``omnivoice.training.trainer``.

"""

logger = logging.getLogger(__name__)

_AUTOCAST_FLEX_ATTENTION = "omnivoice_flex_attention"


def _autocast_flex_attention(module, query, key, value, *args, **kwargs):
    """flex_attention with the same autocast treatment SDPA already gets.

    Mixed-precision training keeps fp32 master weights, so Qwen3's
    ``q_norm``/``k_norm`` (fp32 weight x bf16 activation) silently promote
    q/k — and, through the fp32 RoPE constants, v — back to fp32. SDPA is
    on autocast's cast list and is downcast at the kernel boundary;
    ``flex_attention`` is not, so with ``attn_implementation:
    "flex_attention"`` all attention math runs in fp32: the fp32 backward
    template is ~12x slower at head_dim=128 (61.8ms vs 5.1ms per
    layer-call, H100, identical mask/shape/layout), and the flex and sdpa
    paths become numerically inconsistent with each other. Casting here
    restores the treatment autocast applies to every other matmul; softmax
    accumulation inside the kernel is fp32 either way.
    """
    if torch.is_autocast_enabled(query.device.type) and query.dtype == torch.float32:
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (t.to(dtype) for t in (query, key, value))
    return ALL_ATTENTION_FUNCTIONS["flex_attention"](
        module, query, key, value, *args, **kwargs
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


_VOICE_CLONE_PROMPT_FORMAT_VERSION = 1


@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor  # (C, T)
    ref_text: str
    ref_rms: float

    def save(self, path: str) -> None:
        """Save this prompt to ``path`` for reuse in a later session."""
        torch.save(
            {
                "format_version": _VOICE_CLONE_PROMPT_FORMAT_VERSION,
                "ref_audio_tokens": self.ref_audio_tokens.detach().cpu(),
                "ref_text": self.ref_text,
                "ref_rms": float(self.ref_rms),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "VoiceClonePrompt":
        """Load a prompt saved with :meth:`save`.

        The returned prompt can be passed directly to
        :meth:`OmniVoice.generate`; the audio tokens are moved to the model
        device automatically during generation, so no manual ``.to(device)``
        is needed.

        Args:
            path: File path previously written by :meth:`save`.
            map_location: Device to load the audio tokens onto.
        Returns:
            The restored :class:`VoiceClonePrompt`.
        """
        data = torch.load(path, map_location=map_location, weights_only=True)
        version = data.get("format_version")
        if version != _VOICE_CLONE_PROMPT_FORMAT_VERSION:
            raise ValueError(f"Unsupported VoiceClonePrompt format version: {version}")
        return cls(
            ref_audio_tokens=data["ref_audio_tokens"],
            ref_text=data["ref_text"],
            ref_rms=data["ref_rms"],
        )


@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class GenerationTask:
    batch_size: int
    texts: List[str]
    target_lens: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]
    speed: Optional[List[float]] = None

    def get_indices(self, config: OmniVoiceGenerationConfig, frame_rate: int):
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: List[int]):
        if not indices:
            return None
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_rms=[self.ref_rms[i] for i in indices],
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Config & Model
# ---------------------------------------------------------------------------


class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: Optional[list[float]] = None,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):
        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        self.audio_codebook_weights = audio_codebook_weights


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig, llm: Optional[PreTrainedModel] = None):
        super().__init__(config)

        if llm is not None:
            # If an LLM instance is provided, use it directly
            # (skipping config-based init).
            self.llm = llm
        else:
            # Otherwise, initialize the LLM from the config.
            self.llm = AutoModel.from_config(self.config.llm_config)

        if self.llm.config._attn_implementation == "flex_attention":
            AttentionInterface.register(
                _AUTOCAST_FLEX_ATTENTION, _autocast_flex_attention
            )
            self.llm.set_attn_implementation(_AUTOCAST_FLEX_ATTENTION)

        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            self.config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        self.audio_heads = nn.Linear(
            self.config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        self.normalized_audio_codebook_weights = [
            w / sum(config.audio_codebook_weights)
            for w in config.audio_codebook_weights
        ]

        self.post_init()

        # Inference-only attributes (set by from_pretrained when not in train mode)
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.duration_estimator = None
        self.sampling_rate = None
        self._asr_pipe = None
        self._asr_model_name = "openai/whisper-large-v3-turbo"
        self._asr_device = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        train_mode = kwargs.pop("train", False)
        load_asr = kwargs.pop("load_asr", False)
        asr_model_name = kwargs.pop("asr_model_name", None)
        asr_device = kwargs.pop("asr_device", None)

        # Suppress noisy INFO logs from transformers/huggingface_hub during loading
        _prev_disable = logging.root.manager.disable
        logging.disable(logging.INFO)

        try:
            # Resolve to local path first; download only if not already cached
            resolved_path = _resolve_model_path(pretrained_model_name_or_path)

            model = super().from_pretrained(resolved_path, *args, **kwargs)

            if not train_mode:
                model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)

                audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")

                if not os.path.isdir(audio_tokenizer_path):
                    audio_tokenizer_path = _resolve_model_path(
                        "eustlb/higgs-audio-v2-tokenizer"
                    )

                # higgs-audio-v2-tokenizer does not support MPS
                # (output channels > 65536)
                tokenizer_device = (
                    "cpu" if str(model.device).startswith("mps") else model.device
                )
                model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                    audio_tokenizer_path, device_map=tokenizer_device
                )
                model.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    audio_tokenizer_path
                )

                model.sampling_rate = model.feature_extractor.sampling_rate

                model.duration_estimator = RuleDurationEstimator()

                if asr_model_name is not None:
                    model._asr_model_name = asr_model_name
                if asr_device is not None:
                    model._asr_device = asr_device
                if load_asr:
                    model.load_asr_model()
        finally:
            logging.disable(_prev_disable)

        return model

    # -------------------------------------------------------------------
    # ASR support (optional, for auto-transcription)
    # -------------------------------------------------------------------

    def load_asr_model(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ):
        """Load a Whisper ASR model for reference audio transcription.

        Args:
            model_name: HuggingFace model name or local path for the Whisper
                model. Defaults to the ``asr_model_name`` passed to
                :meth:`from_pretrained` (``openai/whisper-large-v3-turbo``
                if unset).
            device: Device to load the ASR model on (e.g. ``"cuda:1"`` or
                ``"cpu"``). Defaults to the ``asr_device`` passed to
                :meth:`from_pretrained`, falling back to the main model's
                device (its first shard when sharded across GPUs).
        """
        from transformers import pipeline as hf_pipeline

        if model_name is None:
            model_name = self._asr_model_name
        if device is None:
            device = self._asr_device if self._asr_device is not None else self.device

        logger.info("Loading ASR model %s ...", model_name)
        asr_dtype = (
            torch.float16 if str(device).startswith(("cuda", "xpu")) else torch.float32
        )

        model_name = _resolve_model_path(model_name)

        # Use `device=` (single-device placement) rather than `device_map=`:
        # pipeline() ignores a plain device string in `device_map` and
        # auto-selects an accelerator, which is why the ASR model could not
        # be moved off the default GPU (#180).
        self._asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=asr_dtype,
            device=device,
        )
        logger.info("ASR model loaded on %s.", device)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, tuple],
    ) -> str:
        """Transcribe audio using the loaded Whisper ASR model.

        Args:
            audio: File path or ``(waveform, sample_rate)`` tuple.
                Waveform can be a numpy array or torch.Tensor of shape
                ``(1, T)`` or ``(T,)``.

        Returns:
            Transcribed text.
        """
        if self._asr_pipe is None:
            raise RuntimeError(
                "ASR model is not loaded. Call model.load_asr_model() first."
            )

        if isinstance(audio, str):
            return self._asr_pipe(audio)["text"].strip()
        else:
            waveform, sr = audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            waveform = np.squeeze(waveform)  # (1, T) or (T,) → (T,)
            audio_input = {
                "array": waveform,
                "sampling_rate": sr,
            }
            return self._asr_pipe(audio_input)["text"].strip()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
        Embedding shape is (batch_size, seq_length, hidden_size).
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

        # Apply shift to audio IDs based on codebook layer
        # audio_ids: [Batch, 8, Seq]
        # codebook_layer_offsets: [1, 8, 1]
        # Result: Layer 0 ID Layer 1 ID + Layer 2 ID + 2050...
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)

        # input: [Batch, 8, Seq] -> output: [Batch, Seq, Hidden]
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        if attention_mask is None and document_ids is not None:
            if not _flex_attention_available:
                raise RuntimeError(
                    "flex_attention is not available in the current environment. "
                    "If you do not need flex_attention, set "
                    '"attn_implementation": "sdpa" in your training config.'
                )
            attention_mask = create_block_mask(
                _get_packed_mask(
                    document_ids[0].to(inputs_embeds.device),
                ),
                B=None,
                H=None,
                Q_LEN=input_ids.size(-1),
                KV_LEN=input_ids.size(-1),
                _compile=True,
                device=inputs_embeds.device,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]

        loss = None

        # Shape: [B, S, C * Vocab]
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        if labels is not None:
            # audio_logits.permute(0, 3, 1, 2):
            # [Batch, Layer, Seq, Vocab] -> [Batch, Vocab, Layer, Seq]
            # per_token_loss shape: [Batch, Layer, Seq]，ignore -100
            per_token_loss = torch.nn.functional.cross_entropy(
                audio_logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            # valid_mask shape: [Batch, Layer, Seq]
            valid_mask = (labels != -100).float()

            # layer_means shape: [num_layers]
            layer_means = (per_token_loss * valid_mask).sum(
                dim=(0, 2)
            ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)

            weights = torch.tensor(
                self.normalized_audio_codebook_weights, device=audio_logits.device
            )
            loss = (layer_means * weights).sum()

        return OmniVoiceModelOutput(
            loss=loss,
            logits=audio_logits,
        )

    def supported_language_ids(self) -> set[str]:
        """Return a list of supported language IDs."""
        return LANG_IDS

    def supported_language_names(self) -> set[str]:
        """Return a list of supported language names."""
        return LANG_NAMES

    # -------------------------------------------------------------------
    # Inference API
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,
        instruct: Union[str, list[str], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
        speed: Union[float, list[Optional[float]], None] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        normalize_text: bool = True,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate speech audio given text in various modes.

        Supports three modes:

        1. **Voice clone** — clone the voice style from the reference audio.
            Should provide ``voice_clone_prompt`` (from
           :meth:`create_voice_clone_prompt`) or ``ref_text`` + ``ref_audio``.
        2. **Voice design** — provide ``instruct`` text describing
           the desired voice style; no reference audio needed.
        3. **Auto** — provide neither; the model picks a voice itself.

        Args:
            text: Target text (single string or list for batch).
            language: Language name (e.g. ``"English"``) or code
                (e.g. ``"en"``). ``None`` for language-agnostic mode.
                Performance is slightly better if you specify the language.
            ref_text: Optional reference text for voice cloning mode.
            ref_audio: Optional reference audio for voice cloning mode.
                Can be a file path or a (waveform, sample_rate) tuple.
            voice_clone_prompt: Reusable prompt from :meth:`create_voice_clone_prompt`
                or :meth:`VoiceClonePrompt.load`.
                If provided, it overrides ``ref_text`` and ``ref_audio``.
            instruct: Style instruction for voice design mode.
            duration: Fixed output duration in seconds. If a single float,
                applies to all items; if a list, one value per item.
                ``None`` (default) lets the model estimate duration from text.
                Overrides ``speed`` when both are provided.
            speed: Speaking speed factor. ``> 1.0`` for faster, ``< 1.0`` for
                slower. If a list, one value per item. ``None`` (default) uses
                the model's default estimation.
            normalize_text: If ``True``, run text normalization on the target
                text before synthesis (numbers, dates, currency, etc. are
                converted to their spoken form, e.g. ``"2345"`` ->
                ``"twenty three forty five"``). Default ``True`` (matching the current service
                configuration). Chinese/English require the
                optional WeTextProcessing dependency; other
                languages use ``num2words`` for bare integers when installed.
                Inline control syntax (``[laughter]``, ``[B EY1 S]``, pinyin
                tone markers) is preserved. See :func:`normalize_text`.
            generation_config: Explicit config object. If provided, takes
                precedence over ``**kwargs``.
            **kwargs: Generation config or its fields:
                denoise: Whether to prepend the ``<|denoise|>`` token.
                num_step: Number of iterative decoding steps.
                guidance_scale: Classifier-free guidance scale.
                t_shift: Time-step shift (smaller → emphasise low-SNR).
                postprocess_output: Post-process output (remove silence, fade-in/out, pad edges).
                layer_penalty_factor: Penalty encouraging earlier codebook
                    layers to unmask first.
                position_temperature: Temperature for position selection.
                class_temperature: Temperature for token sampling (0 = greedy).
                audio_chunk_duration: If > 0, split long text into chunks of
                    this duration (seconds) and generate chunk by chunk.
                audio_chunk_threshold: Only apply chunking if estimated audio
                    duration exceeds this threshold (seconds).
                pad_duration: Silence padding duration per side in seconds
                    (0 to disable).
                fade_duration: Fade-in/out curve duration in seconds
                    (0 to disable).
        Returns:
            ``audios`` a list of 1-D ``np.ndarray`` with shape ``(T,)`` and
            sampling rate consistent with the model's audio tokenizer
            (usually 24 000 Hz).  Can be saved directly with
            ``soundfile.write("out.wav", audios[0], model.sampling_rate)``.
        """

        if self.audio_tokenizer is None or self.text_tokenizer is None:
            raise RuntimeError(
                "Model is not loaded with audio/text tokenizers. Make sure you "
                "loaded the model with OmniVoice.from_pretrained()."
            )
        gen_config = (
            generation_config
            if generation_config is not None
            else OmniVoiceGenerationConfig.from_dict(kwargs)
        )

        self.eval()

        full_task = self._preprocess_all(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=speed,
            duration=duration,
            normalize_text=normalize_text,
        )

        short_idx, long_idx = full_task.get_indices(
            gen_config, self.audio_tokenizer.config.frame_rate
        )

        results = [None] * full_task.batch_size

        if short_idx:
            short_task = full_task.slice_task(short_idx)
            short_results = self._generate_iterative(short_task, gen_config)
            for idx, res in zip(short_idx, short_results):
                results[idx] = res

        if long_idx:
            long_task = full_task.slice_task(long_idx)
            long_results = self._generate_chunked(long_task, gen_config)
            for idx, res in zip(long_idx, long_results):
                results[idx] = res

        generated_audios = []
        for i in range(full_task.batch_size):
            assert results[i] is not None, f"Result {i} was not generated"
            generated_audios.append(
                self._decode_and_post_process(
                    results[i],
                    full_task.ref_rms[i],
                    gen_config,  # type: ignore[arg-type]
                )
            )

        return generated_audios

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> VoiceClonePrompt:
        """Create a reusable voice clone prompt from reference audio.

        Args:
            ref_audio: File path (str) or ``(waveform, sample_rate)`` tuple.
                waveform should be a 1-D or 2-D torch.Tensor (channels x samples).
            ref_text: Transcript of the reference audio. If ``None``, the
                ASR model will be used to auto-transcribe (must call
                :meth:`load_asr_model` first).
            preprocess_prompt: If ``True`` (default), apply silence removal and
                trimming to the reference audio, add punctuation in the end
                of reference text (if not already)

        Returns:
            A :class:`VoiceClonePrompt` that can be passed to :meth:`generate`.
        """
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "Audio tokenizer is not loaded. Make sure you loaded the model "
                "with OmniVoice.from_pretrained()."
            )

        if isinstance(ref_audio, str):
            ref_wav = load_audio(ref_audio, self.sampling_rate)
        else:
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0, keepdims=True)
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform),
                    orig_freq=sr,
                    new_freq=self.sampling_rate,
                ).numpy()
            ref_wav = waveform

        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        if preprocess_prompt:
            # Trim long reference audio (>20s) by splitting at the largest silence gap.
            # Skip trimming when ref_text is user-provided, otherwise the
            # trimmed audio will no longer match the full transcript.
            if ref_text is None:
                ref_wav = trim_long_audio(
                    ref_wav, self.sampling_rate, trim_threshold=20.0
                )
            ref_wav = remove_silence(
                ref_wav,
                self.sampling_rate,
                mid_sil=200,
                lead_sil=100,
                trail_sil=200,
            )
            if ref_wav.shape[-1] == 0:
                raise ValueError(
                    "Reference audio is empty after silence removal. "
                    "Try setting preprocess_prompt=False."
                )

        ref_duration = ref_wav.shape[-1] / self.sampling_rate
        if ref_duration > 20.0:
            logger.warning(
                "Reference audio is %.1fs long (>20s). This may cause slower "
                "generation, higher memory usage, and degraded voice cloning "
                "quality. We recommend trimming it to 3-10s.",
                ref_duration,
            )

        # Auto-transcribe if ref_text not provided
        if ref_text is None:
            if self._asr_pipe is None:
                logger.info("ASR model not loaded yet, loading on-the-fly ...")
                self.load_asr_model()
            ref_text = self.transcribe((ref_wav, self.sampling_rate))
            logger.debug("Auto-transcribed ref_text: %s", ref_text)

        chunk_size = self.audio_tokenizer.config.hop_length
        clip_size = int(ref_wav.shape[-1] % chunk_size)
        ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
        # numpy → torch at tokenizer boundary
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        ref_audio_tokens = self.audio_tokenizer.encode(
            ref_wav_tensor.unsqueeze(0),
        ).audio_codes.squeeze(0)  # (C, T)

        if preprocess_prompt:
            ref_text = add_punctuation(ref_text)

        return VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=ref_text,
            ref_rms=ref_rms,
        )

    def _decode_and_post_process(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
        rms: Union[float, None],
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        """
        Args:
            tokens: Audio tokens — either a single tensor of shape
                (num_codebooks, seq_len) or a list of chunk tensors.
            rms: RMS of the reference audio for volume adjustment.
            gen_config: Generation config for post-processing options.
        Returns:
            Decoded and post-processed audio array of shape (T,).
        """
        tokenizer_device = self.audio_tokenizer.device
        if isinstance(tokens, list):
            chunk_audios = [
                self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
                for t in tokens
            ]
            audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            audio_waveform = (
                self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
            )

        audio_waveform = self._post_process_audio(
            audio_waveform,
            ref_rms=rms,
            gen_config=gen_config,
        )
        return audio_waveform.squeeze(0)

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        ref_rms: Union[float, None],
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        """Optionally remove long silences, adjust volume, and add edge padding.

        Args:
            generated_audio: Numpy array of shape (1, T).
            ref_rms: RMS of the reference audio for volume normalisation.
            gen_config: Generation config controlling post-processing behaviour.
        Returns:
            Processed numpy array of shape (1, T).
        """
        if gen_config.postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        if ref_rms is not None and ref_rms < 0.1:
            generated_audio = generated_audio * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(generated_audio).max()
            if peak > 1e-6:
                generated_audio = generated_audio / peak * 0.5

        generated_audio = fade_and_pad_audio(
            generated_audio,
            pad_duration=gen_config.pad_duration,
            fade_duration=gen_config.fade_duration,
            sample_rate=self.sampling_rate,
        )
        return generated_audio

    def _generate_chunked(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[List[torch.Tensor]]:
        """Generate long audio by splitting text into chunks and batching.

        Each item in the returned list corresponds to one input and contains
        a list of audio token tensors — one per text chunk.

        Args:
            task: A :class:`GenerationTask` with one or more items whose
                estimated audio exceeds ``audio_chunk_threshold``.
            gen_config: Generation config (``audio_chunk_duration`` controls
                chunk size).
        Returns:
            Per-item list of chunk token-tensor lists.
        """
        # Chunk each item's text
        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration
                * self.audio_tokenizer.config.frame_rate
                / avg_tokens_per_char
            )
            chunks = chunk_text_punctuation(
                text=task.texts[i],
                chunk_len=text_chunk_len,
                min_chunk_len=3,
            )
            logger.debug(f"Item {i} chunked into {len(chunks)} pieces: {chunks}")
            all_chunks.append(chunks)

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have "
            "ref_audio. Mixed ref/non-ref is not supported."
        )

        max_num_chunks = max(len(c) for c in all_chunks)

        # chunk_results[item_idx] = list of generated token tensors per chunk
        chunk_results = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            # All items have reference audio.
            # We still sequentially generate chunks within each item, but we
            # batch across items for the same chunk index. This allows to keep
            # the VRAM usage manageable while still benefiting from batching.
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[task.ref_audio_tokens[i] for i in indices],
                    ref_texts=[task.ref_texts[i] for i in indices],
                )
        else:
            # No reference audio — generate chunk 0 for all items first,
            # then use chunk 0 output as reference for all subsequent chunks.
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}

            # Batch all remaining chunks, using chunk 0 as fixed reference
            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[first_chunk_map[i] for i in indices],
                    ref_texts=[all_chunks[i][0] for i in indices],
                )

        return chunk_results

    def _preprocess_all(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,
        instruct: Union[str, list[str], None] = None,
        preprocess_prompt: bool = True,
        speed: Union[float, list[Optional[float]], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
        normalize_text: bool = True,
    ) -> GenerationTask:
        if isinstance(text, str):
            text_list = [text]
        else:
            assert isinstance(text, list), (
                "text should be a string or a list of strings"
            )
            text_list = text
        batch_size = len(text_list)

        language_list = self._ensure_list(language, batch_size)
        language_list = [_resolve_language(lang) for lang in language_list]

        # Optional text normalization (opt-in). Applied to the target text only
        # (not ref_text, which must stay aligned with the reference audio),
        # before duration estimation so the estimate matches the spoken form.
        if normalize_text:
            text_list = [
                _normalize_text(t, lang) for t, lang in zip(text_list, language_list)
            ]
        instruct_list = self._ensure_list(instruct, batch_size)
        for i, s in enumerate(instruct_list):
            if s is None:
                continue
            use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i]))
            instruct_list[i] = _resolve_instruct(s, use_zh=use_zh)

        if voice_clone_prompt is not None and (
            ref_text is not None or ref_audio is not None
        ):
            logger.warning(
                "Both voice_clone_prompt and ref_text/ref_audio are provided. "
                "ref_text/ref_audio will be ignored."
            )
        if voice_clone_prompt is None and ref_audio is not None:
            # If voice_clone_prompt is not provided, create it from
            # ref_audio (ref_text will be auto-transcribed if not given).
            ref_text_list = self._ensure_list(ref_text, batch_size, auto_repeat=False)
            ref_audio_list = self._ensure_list(ref_audio, batch_size, auto_repeat=False)

            voice_clone_prompt = []
            for i in range(len(ref_text_list)):
                voice_clone_prompt.append(
                    self.create_voice_clone_prompt(
                        ref_audio=ref_audio_list[i],
                        ref_text=ref_text_list[i],
                        preprocess_prompt=preprocess_prompt,
                    )
                )

        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)
        if voice_clone_prompt_list[0] is not None:
            ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
            ref_audio_tokens_list = [
                vc.ref_audio_tokens for vc in voice_clone_prompt_list
            ]
            ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]
        else:
            ref_text_list = [None] * batch_size
            ref_audio_tokens_list = [None] * batch_size
            ref_rms_list = [None] * batch_size

        # Normalize speed/duration to per-item lists (may contain None).
        if speed is not None:
            if isinstance(speed, (int, float)):
                user_speed = [float(speed)] * batch_size
            else:
                user_speed = list(speed)
        else:
            user_speed = None

        if duration is not None:
            if isinstance(duration, (int, float)):
                durations = [float(duration)] * batch_size
            else:
                durations = list(duration)
        else:
            durations = None

        num_target_tokens_list = []
        for i in range(batch_size):
            # duration[i] overrides speed for estimation: use speed=1.0
            # to get the raw estimate, then override target_lens below.
            has_dur = durations is not None and durations[i] is not None
            item_speed = 1.0 if has_dur else (user_speed[i] if user_speed else 1.0)
            est = self._estimate_target_tokens(
                text_list[i],
                ref_text_list[i],
                ref_audio_tokens_list[i].size(-1)
                if ref_audio_tokens_list[i] is not None
                else None,
                speed=item_speed,
            )
            num_target_tokens_list.append(est)

        # Per-item duration overrides: set target_lens to exact frame count
        # and compute speed ratio so chunked generation scales proportionally.
        speed_list: Optional[List[float]] = None
        if durations is not None:
            frame_rate = self.audio_tokenizer.config.frame_rate
            speed_list = []
            for i in range(batch_size):
                if durations[i] is not None:
                    target_tokens = max(1, int(durations[i] * frame_rate))
                    est = num_target_tokens_list[i]
                    speed_list.append(est / target_tokens if target_tokens > 0 else 1.0)
                    num_target_tokens_list[i] = target_tokens
                else:
                    s = user_speed[i] if user_speed else None
                    speed_list.append(s if s is not None else 1.0)
        elif user_speed is not None:
            speed_list = [s if s is not None else 1.0 for s in user_speed]

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=num_target_tokens_list,
            langs=language_list,
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_tokens_list,
            ref_rms=ref_rms_list,
            speed=speed_list,
        )

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        """Estimate number of target audio tokens."""
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            # Fall back to a simple heuristic
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25

        est = self.duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _ensure_list(
        self, x: Union[Any, List[Any]], batch_size: int, auto_repeat: bool = True
    ) -> List[Any]:
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in (
            1,
            batch_size,
        ):
            raise ValueError(
                f"should be either the number of the text or 1, but got {len(x_list)}"
            )
        if auto_repeat and len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
    ):
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                with shape (C, T).
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.
        """

        # Build style tokens: <|denoise|> + <|lang_start|>...<|lang_end|>
        #                      + <|instruct_start|>...<|instruct_end|>
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(self.device)  # [1, C, N1]

        # Build text tokens
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(self.device)  # [1, C, N2]

        # Target: all MASK
        target_audio_tokens = torch.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # Conditional input
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(
            1, cond_total_length, dtype=torch.bool, device=self.device
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_iterative(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[torch.Tensor]:
        """N-step iterative unmasked decoding.

        Args:
            task: A :class:`GenerationTask` containing batch texts, target
                lengths, languages, instructions, and optional reference data.
            gen_config: A :class:`OmniVoiceGenerationConfig` controlling
                decoding steps, guidance, temperatures, etc.
        Returns:
            List of generated audio token tensors of shape (C, T) (one per
            input text).
        """

        B = task.batch_size

        for i in range(B):
            logger.debug(
                "Item %d — text: %s | ref_text: %s | instruct: %s | lang: %s | target_tokens: %d",
                i,
                task.texts[i],
                task.ref_texts[i],
                task.instructs[i],
                task.langs[i],
                task.target_lens[i],
            )

        inputs_list = [
            self._prepare_inference_inputs(
                task.texts[i],
                task.target_lens[i],
                task.ref_texts[i],
                task.ref_audio_tokens[i],
                task.langs[i],
                task.instructs[i],
                gen_config.denoise,
            )
            for i in range(B)
        ]

        c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id  # Or any other tokens

        batch_input_ids = torch.full(
            (2 * B, self.config.num_audio_codebook, max_c_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_audio_mask = torch.zeros(
            (2 * B, max_c_len), dtype=torch.bool, device=self.device
        )
        batch_attention_mask = torch.zeros(
            (2 * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device
        )

        for i, inp in enumerate(inputs_list):
            c_len, u_len = c_lens[i], task.target_lens[i]

            # Cond (0 ~ B-1)
            batch_input_ids[i, :, :c_len] = inp["input_ids"]
            batch_audio_mask[i, :c_len] = inp["audio_mask"]
            batch_attention_mask[i, :, :c_len, :c_len] = True

            # Uncond (B ~ 2B-1)
            batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[B + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (B, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(
            self.config.num_audio_codebook, device=self.device
        ).view(1, -1, 1)

        for step in range(gen_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len, t_len = c_lens[i], task.target_lens[i]

                # Extract real target Logits
                # [1, C, T, V]
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]

                pred_tokens, scores = self._predict_tokens_with_scoring(
                    c_logits, u_logits, gen_config
                )

                scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(
                    sample_tokens != self.config.audio_mask_id, -float("inf")
                )

                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # Update individual slices into batched structure
                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[B + i : B + i + 1, :, :t_len] = sample_tokens

        return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(
                filtered_probs, gen_config.class_temperature
            ).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]

        return pred_tokens, confidence_scores


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _get_packed_mask(document_ids):
    return partial(_mask_mod_packed, document_ids)


def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    # 1. Sequence Packing Logic: Tokens must belong to the same document.
    # Note: The doc_id for padding tokens is -1, which will automatically not match
    # (if handled correctly) or be ignored.
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    return same_doc


def _resolve_language(language: Optional[str]) -> Union[str, None]:
    pass

    if language is None or language.lower() == "none":
        return None
    if language in LANG_IDS:
        return language
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    logger.warning(
        f"Language '{language}' is not recognized. "
        f"Please use a valid language ID (e.g., 'en', 'zh', 'ja', 'de') "
        f"or a full language name (e.g., 'English', 'Chinese', 'Japanese'). "
        f"See supported_language_ids() or supported_language_names() for details. "
        f"Falling back to None (language-agnostic mode)."
    )
    return None


def _resolve_instruct(
    instruct: Optional[str], use_zh: bool = False
) -> Union[str, None]:
    """Validate and normalise a voice-design instruct string.

    Supported instruct items (case-insensitive for English):

    English (comma + space separated):
        gender: male, female
        age: child, teenager, young adult, middle-aged, elderly
        pitch: very low pitch, low pitch, moderate pitch,
               high pitch, very high pitch
        style: whisper
        accent: american accent, british accent, australian accent, ...

    Chinese (full-width comma separated):
        gender: 男, 女
        age: 儿童, 少年, 青年, 中年, 老年
        pitch: 极低音调, 低音调, 中音调, 高音调, 极高音调
        style: 耳语
        dialect: 河南话, 陕西话, 四川话, 贵州话, 云南话,
                 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话,
                 青岛话, 东北话

    Minor issues (auto-fixed):
      - Wrong separator (half-width comma in Chinese instruct or
        full-width comma in English instruct)
      - Leading / trailing commas

    Major issues (raise ``ValueError``):
      - Unsupported or misspelled instruct items
      - Suggestions are offered for close matches

    Args:
        instruct: Raw instruct string, or ``None``.
        use_zh: If True, normalise all items to Chinese (used when the
            synthesis text contains Chinese and no accent is specified).

    Returns:
        Normalised instruct string, or ``None``.

    Raises:
        ValueError: if any instruct item is unsupported or misspelled.
    """
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    # Split on both half-width and full-width commas
    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]

    # Validate each item
    unknown = []
    normalised = []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append(f"  '{raw}' -> '{n}' (unsupported; did you mean '{sug}'?)")
            else:
                lines.append(f"  '{raw}' -> '{n}' (unsupported)")
        err = (
            f"Unsupported instruct items found in {instruct_str}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )
        raise ValueError(err)

    # --- Language consistency: dialect forces Chinese, accent forces English ---
    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)

    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    # --- Unify to single language ---
    if use_zh:
        normalised = [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalised]
    else:
        normalised = [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalised]

    # --- Category conflict check ---
    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join(f"'{x}'" for x in group))
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item."
        )

    # Determine separator based on language
    has_zh = any(any("\u4e00" <= c <= "\u9fff" for c in n) for n in normalised)
    separator = "，" if has_zh else ", "

    return separator.join(normalised)


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        result = tokenizer(text, return_tensors="pt").input_ids
    else:
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result


def _combine_text(text, ref_text: Optional[str] = None) -> str:
    # combine with reference text if not None
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()

    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text


# ---------------------------------------------------------------------------
# Register with HuggingFace Auto classes
# ---------------------------------------------------------------------------

AutoConfig.register("omnivoice", OmniVoiceConfig, exist_ok=True)
AutoModel.register(OmniVoiceConfig, OmniVoice, exist_ok=True)

__all__ = [
    "OmniVoice",
    "OmniVoiceConfig",
    "OmniVoiceGenerationConfig",
    "VoiceClonePrompt",
]
