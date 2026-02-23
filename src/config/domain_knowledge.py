"""Static domain knowledge maps for rave and electronic music culture.

# ─── PURPOSE (Junior Developer Guide) ──────────────────────────────────
#
# This module contains curated, hand-coded knowledge about electronic music
# culture that is used to *boost* corpus search result quality.  When a user
# queries the RAG pipeline for "detroit techno 1992", the helpers here let
# the pipeline understand that:
#
#   - "detroit techno" is adjacent to "electro", "techno", "deep house", etc.
#   - Detroit, Michigan, and Belleville are relevant geographic locations.
#   - 1992 falls within the "early techno" temporal window (1988-1992).
#   - Artists like Juan Atkins, Derrick May, and Jeff Mills are core to
#     that scene, and each has side projects / aliases the user might
#     reference instead of their canonical name.
#
# All functions are **pure** (no side effects, no I/O, no external deps).
# Data structures are built once at module-load time and thereafter
# accessed in O(1) dict/set lookups.
# ──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re


# ═════════════════════════════════════════════════════════════════════════
# 1. GENRE ADJACENCY MAP
# ═════════════════════════════════════════════════════════════════════════
# Maps a genre name (lowercase) to the set of genres considered "adjacent"
# — meaning a flier mentioning genre A is likely relevant to a query about
# genre B if B is in A's adjacency set and vice-versa.

GENRE_ADJACENCY: dict[str, set[str]] = {
    "techno": {
        "detroit techno", "minimal techno", "industrial techno",
        "acid techno", "hard techno", "dub techno", "electro",
        "ambient techno", "tech house", "ebm",
    },
    "detroit techno": {
        "techno", "electro", "deep house", "chicago house",
        "ambient techno", "minimal techno",
    },
    "minimal techno": {
        "techno", "dub techno", "ambient techno", "tech house",
        "microhouse",
    },
    "industrial techno": {
        "techno", "hard techno", "industrial", "ebm",
        "acid techno", "noise",
    },
    "acid techno": {
        "techno", "acid house", "hard techno", "industrial techno",
        "breakbeat hardcore",
    },
    "hard techno": {
        "techno", "industrial techno", "acid techno", "hardcore",
        "gabber", "schranz",
    },
    "dub techno": {
        "techno", "minimal techno", "ambient techno", "dub",
        "deep house",
    },
    "house": {
        "deep house", "acid house", "chicago house", "progressive house",
        "tech house", "soulful house", "garage", "uk garage",
        "disco", "uk funky",
    },
    "deep house": {
        "house", "soulful house", "tech house", "chicago house",
        "detroit techno", "garage", "dub techno",
    },
    "acid house": {
        "house", "chicago house", "acid techno", "techno",
        "breakbeat hardcore", "rave",
    },
    "chicago house": {
        "house", "acid house", "deep house", "soulful house",
        "detroit techno", "disco", "garage",
    },
    "progressive house": {
        "house", "trance", "progressive trance", "tech house",
        "deep house",
    },
    "tech house": {
        "house", "techno", "minimal techno", "deep house",
        "progressive house", "microhouse",
    },
    "soulful house": {
        "house", "deep house", "chicago house", "garage",
        "disco", "gospel house",
    },
    "jungle": {
        "drum and bass", "breakbeat hardcore", "ragga jungle",
        "breakbeat", "hardcore", "reggae", "dub",
    },
    "drum and bass": {
        "jungle", "liquid dnb", "neurofunk", "breakbeat",
        "breakbeat hardcore", "techstep",
    },
    "liquid dnb": {
        "drum and bass", "jungle", "soulful house", "ambient",
    },
    "neurofunk": {
        "drum and bass", "techstep", "jungle",
    },
    "breakbeat": {
        "breakbeat hardcore", "big beat", "jungle",
        "drum and bass", "nu skool breaks",
    },
    "breakbeat hardcore": {
        "breakbeat", "jungle", "hardcore", "rave",
        "acid house", "happy hardcore",
    },
    "big beat": {
        "breakbeat", "electro", "rock", "hip hop",
        "nu skool breaks",
    },
    "trance": {
        "progressive trance", "goa trance", "psytrance",
        "acid trance", "progressive house", "hard trance",
        "uplifting trance",
    },
    "progressive trance": {
        "trance", "progressive house", "goa trance",
        "psytrance",
    },
    "goa trance": {
        "trance", "psytrance", "progressive trance",
        "acid trance", "full on",
    },
    "psytrance": {
        "goa trance", "trance", "progressive trance",
        "darkpsy", "full on", "forest",
    },
    "acid trance": {
        "trance", "acid house", "acid techno", "goa trance",
    },
    "garage": {
        "uk garage", "speed garage", "2-step", "house",
        "deep house", "soulful house", "r&b",
    },
    "uk garage": {
        "garage", "speed garage", "2-step", "grime",
        "dubstep", "house", "uk funky", "bassline",
    },
    "speed garage": {
        "garage", "uk garage", "2-step", "house",
        "bassline",
    },
    "2-step": {
        "uk garage", "garage", "speed garage", "grime",
        "dubstep", "r&b",
    },
    "dubstep": {
        "grime", "uk garage", "2-step", "bass music",
        "dub", "drum and bass",
    },
    "grime": {
        "uk garage", "2-step", "dubstep", "bass music",
        "hip hop", "dancehall",
    },
    "bass music": {
        "dubstep", "grime", "uk garage", "drum and bass",
        "uk funky", "bassline", "trap",
    },
    "ambient": {
        "ambient techno", "downtempo", "chill out", "idm",
        "drone", "new age",
    },
    "ambient techno": {
        "ambient", "dub techno", "techno", "downtempo",
        "minimal techno", "idm",
    },
    "downtempo": {
        "ambient", "chill out", "trip hop", "idm",
        "lounge",
    },
    "chill out": {
        "ambient", "downtempo", "lounge", "balearic",
        "trip hop",
    },
    "idm": {
        "ambient", "ambient techno", "glitch", "electro",
        "downtempo", "breakbeat",
    },
    "hardcore": {
        "gabber", "happy hardcore", "hard techno",
        "breakbeat hardcore", "industrial", "speedcore",
    },
    "gabber": {
        "hardcore", "hard techno", "speedcore",
        "industrial", "terrorcore",
    },
    "happy hardcore": {
        "hardcore", "breakbeat hardcore", "uk hardcore",
        "trance", "makina",
    },
    "industrial": {
        "ebm", "industrial techno", "noise", "techno",
        "power electronics",
    },
    "ebm": {
        "industrial", "techno", "industrial techno",
        "synth pop", "dark electro",
    },
    "electro": {
        "detroit techno", "techno", "miami bass",
        "freestyle", "hip hop", "idm",
    },
    "miami bass": {
        "electro", "freestyle", "hip hop", "bass music",
        "booty bass",
    },
    "freestyle": {
        "electro", "miami bass", "synth pop", "disco",
        "house",
    },
    "uk funky": {
        "uk garage", "house", "afrobeats", "soca",
        "bass music", "funky house",
    },
    "footwork": {
        "juke", "ghetto house", "chicago house",
        "bass music", "hip hop",
    },
    "juke": {
        "footwork", "ghetto house", "chicago house",
        "bass music",
    },
    "trip hop": {
        "downtempo", "chill out", "hip hop", "ambient",
        "dub",
    },
    "dub": {
        "dub techno", "reggae", "jungle", "dubstep",
        "trip hop",
    },
    "disco": {
        "house", "chicago house", "soulful house",
        "nu disco", "italo disco", "boogie",
    },
    "hard trance": {
        "trance", "hard techno", "hardcore", "gabber",
        "uplifting trance",
    },
    "uk hardcore": {
        "happy hardcore", "hardcore", "makina",
        "breakbeat hardcore",
    },
    "techstep": {
        "drum and bass", "neurofunk", "jungle",
    },
    "ragga jungle": {
        "jungle", "drum and bass", "reggae", "dancehall",
    },
    "microhouse": {
        "minimal techno", "tech house", "deep house",
        "glitch",
    },
    "schranz": {
        "hard techno", "techno", "industrial techno",
    },
    "noise": {
        "industrial", "power electronics", "harsh noise wall",
    },
    "bassline": {
        "uk garage", "speed garage", "bass music",
        "uk funky",
    },
    "nu skool breaks": {
        "breakbeat", "big beat", "progressive house",
    },
    "glitch": {
        "idm", "microhouse", "ambient", "experimental",
    },
    "balearic": {
        "chill out", "house", "disco", "new wave",
    },
}


def get_adjacent_genres(genre: str) -> set[str]:
    """Return genres adjacent to the given genre (case-insensitive).

    Returns an empty set if the genre is not found in the adjacency map.

    Args:
        genre: The genre name to look up.

    Returns:
        A set of adjacent genre name strings, or an empty set.
    """
    return GENRE_ADJACENCY.get(genre.lower().strip(), set())


# ═════════════════════════════════════════════════════════════════════════
# 2. SCENE-TO-GEOGRAPHY MAP
# ═════════════════════════════════════════════════════════════════════════
# Maps scene-specific keywords or phrases to geographic locations strongly
# associated with that scene.  The pipeline uses this to add geographic
# context when a user's query references a cultural scene but does not
# mention a specific city.

SCENE_GEOGRAPHY: dict[str, list[str]] = {
    "warehouse parties": [
        "New York", "Detroit", "London", "Berlin", "Manchester", "Chicago",
    ],
    "acid house": [
        "Chicago", "London", "Manchester", "Sheffield", "Ibiza",
    ],
    "second summer of love": [
        "London", "Manchester", "Sheffield", "Ibiza", "UK",
    ],
    "detroit techno": [
        "Detroit", "Michigan", "Belleville",
    ],
    "chicago house": [
        "Chicago", "Illinois",
    ],
    "berlin techno": [
        "Berlin", "Germany", "Frankfurt",
    ],
    "uk garage": [
        "London", "South London", "East London", "UK",
    ],
    "jungle": [
        "London", "South London", "East London", "Bristol", "UK",
    ],
    "free party": [
        "UK", "France", "Czech Republic", "Italy", "Castlemorton",
    ],
    "sound system culture": [
        "Jamaica", "London", "Bristol", "Nottingham", "UK",
    ],
    "criminal justice act": [
        "UK", "London", "England", "Wales",
    ],
    "rave": [
        "UK", "London", "Manchester", "Berlin", "Detroit",
        "Ibiza", "Goa",
    ],
    "hacienda": [
        "Manchester", "UK",
    ],
    "tresor": [
        "Berlin", "Germany",
    ],
    "berghain": [
        "Berlin", "Germany",
    ],
    "fabric": [
        "London", "Farringdon", "UK",
    ],
    "ministry of sound": [
        "London", "Elephant and Castle", "UK",
    ],
    "paradise garage": [
        "New York", "Manhattan", "NYC",
    ],
    "warehouse project": [
        "Manchester", "UK",
    ],
    "defqon": [
        "Netherlands", "Biddinghuizen", "Australia",
    ],
    "thunderdome": [
        "Netherlands", "Rotterdam", "Amsterdam",
    ],
    "spiral tribe": [
        "UK", "France", "Czech Republic", "Europe",
    ],
    "teknival": [
        "France", "Czech Republic", "Italy", "UK", "Europe",
    ],
    "illegal rave": [
        "UK", "London", "Manchester", "Berlin", "Detroit",
    ],
    "pirate radio": [
        "London", "East London", "UK",
    ],
    "hardcore continuum": [
        "London", "UK",
    ],
    "uk hardcore": [
        "UK", "London", "Scotland",
    ],
    "gabber": [
        "Netherlands", "Rotterdam", "Amsterdam", "Germany",
    ],
    "goa trance": [
        "Goa", "India", "Ibiza", "Israel",
    ],
    "motor city": [
        "Detroit", "Michigan",
    ],
    "the loft": [
        "New York", "Manhattan", "NYC",
    ],
    "boiler room": [
        "London", "Berlin", "New York", "Amsterdam",
    ],
}


def get_scene_geographies(query: str) -> set[str]:
    """Return geographic locations associated with scene keywords found in the query.

    Scans the query text for all matching scene keywords in SCENE_GEOGRAPHY
    and returns the union of their associated geographic locations.

    Args:
        query: The user's search query text.

    Returns:
        A set of geographic location strings, or an empty set if no
        scene keywords are detected.
    """
    query_lower = query.lower()
    locations: set[str] = set()
    for scene_keyword, geos in SCENE_GEOGRAPHY.items():
        if scene_keyword in query_lower:
            locations.update(geos)
    return locations


# ═════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL SIGNAL DETECTION
# ═════════════════════════════════════════════════════════════════════════
# Compiled regex patterns and named-era mappings that extract time-period
# references from natural-language queries.

# Named era -> canonical time-period string.
_NAMED_ERAS: dict[str, str] = {
    "second summer of love": "1988-1989",
    "summer of love": "1988-1989",
    "early rave": "1988-1992",
    "old school": "1988-1995",
    "old skool": "1988-1995",
    "golden era": "1993-1997",
    "golden age": "1993-1997",
    "early techno": "1988-1992",
    "classic house": "1986-1992",
    "classic techno": "1988-1995",
    "classic jungle": "1992-1997",
    "classic drum and bass": "1995-2000",
    "early house": "1984-1990",
    "early jungle": "1991-1994",
    "early dubstep": "2001-2006",
    "early grime": "2002-2006",
    "first wave rave": "1988-1992",
    "second wave rave": "1992-1996",
    "acid era": "1987-1991",
    "uk garage era": "1997-2002",
    "nuum": "1990-2010",
}

# Compiled regex patterns.  Order matters: more specific patterns are
# checked before less specific ones.  The second element of each tuple
# is a "kind" tag that tells detect_temporal_signal() how to interpret
# the match groups.  A kind of None means "look up in _NAMED_ERAS".
_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], str | None]] = [
    # Named eras (checked against _NAMED_ERAS dict).
    (re.compile(
        r"\b(?:" + "|".join(re.escape(era) for era in sorted(
            _NAMED_ERAS, key=len, reverse=True
        )) + r")\b",
        re.IGNORECASE,
    ), None),

    # Explicit year ranges: "1988-1992", "1988 to 1992"
    (re.compile(r"\b((?:19|20)\d{2})\s*[-\u2013\u2014]\s*((?:19|20)\d{2})\b"), "range"),

    # Explicit decades: "1990s", "the 90s", "90's"
    (re.compile(r"\bthe\s+(\d{2})['\u2019]?s\b", re.IGNORECASE), "short_decade"),
    (re.compile(r"\b(\d{2})['\u2019]?s\b"), "short_decade"),
    (re.compile(r"\b((?:19|20)\d{2})s\b"), "full_decade"),

    # Explicit single years: "1992", "2001"
    (re.compile(r"\b((?:19|20)\d{2})\b"), "single_year"),
]


def detect_temporal_signal(query: str) -> str | None:
    """Extract a time period from the query text.

    Checks for named eras first, then explicit date patterns.

    Args:
        query: The user's search query text.

    Returns:
        A decade string (e.g. "1990s"), year range (e.g. "1988-1992"),
        single year (e.g. "1992"), or None if no temporal signal is found.
    """
    query_lower = query.lower().strip()

    # Pass 1: Named eras (longest match first to avoid substring collisions).
    for era_name in sorted(_NAMED_ERAS, key=len, reverse=True):
        if era_name in query_lower:
            return _NAMED_ERAS[era_name]

    # Pass 2: Regex-based patterns.
    for pattern, kind in _TEMPORAL_PATTERNS:
        if kind is None:
            # Named-era regex — already handled in Pass 1.
            continue
        match = pattern.search(query)
        if match:
            if kind == "range":
                return f"{match.group(1)}-{match.group(2)}"
            if kind == "full_decade":
                return f"{match.group(1)}s"
            if kind == "short_decade":
                short = int(match.group(1))
                century = 19 if short >= 50 else 20
                return f"{century}{short:02d}s"
            if kind == "single_year":
                return match.group(1)

    return None


def _parse_period_to_range(period: str) -> tuple[int, int] | None:
    """Convert a period string to (start_year, end_year) inclusive.

    Understands:
      - "1990s"       -> (1990, 1999)
      - "1988-1995"   -> (1988, 1995)
      - "1992"        -> (1992, 1992)

    Returns None if the string cannot be parsed.
    """
    period = period.strip()

    # Decade: "1990s"
    decade_match = re.fullmatch(r"((?:19|20)\d{2})s", period)
    if decade_match:
        start = int(decade_match.group(1))
        return (start, start + 9)

    # Range: "1988-1995"
    range_match = re.fullmatch(
        r"((?:19|20)\d{2})\s*[-\u2013\u2014]\s*((?:19|20)\d{2})", period
    )
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))

    # Single year: "1992"
    year_match = re.fullmatch(r"((?:19|20)\d{2})", period)
    if year_match:
        year = int(year_match.group(1))
        return (year, year)

    return None


def temporal_overlap(period_a: str | None, period_b: str | None) -> bool:
    """Return True if two time period strings overlap.

    Examples:
        temporal_overlap("1990s", "1988-1995")  ->  True   (overlap 1990-1995)
        temporal_overlap("1980s", "1990s")       ->  False
        temporal_overlap("1992", "1990s")        ->  True
        temporal_overlap(None, "1990s")          ->  False

    Args:
        period_a: First period string (decade, range, or year), or None.
        period_b: Second period string (decade, range, or year), or None.

    Returns:
        True if the periods overlap, False otherwise (including when
        either period is None or unparseable).
    """
    if period_a is None or period_b is None:
        return False

    range_a = _parse_period_to_range(period_a)
    range_b = _parse_period_to_range(period_b)

    if range_a is None or range_b is None:
        return False

    # Two ranges [a_start, a_end] and [b_start, b_end] overlap iff:
    #   a_start <= b_end AND b_start <= a_end
    return range_a[0] <= range_b[1] and range_b[0] <= range_a[1]


# ═════════════════════════════════════════════════════════════════════════
# 4. ARTIST ALIAS TABLE
# ═════════════════════════════════════════════════════════════════════════
# Maps canonical artist names to sets of known aliases, side projects, and
# real names.  The inverted index (_ALIAS_TO_CANONICAL) is built once at
# module load time for O(1) reverse lookups.

ARTIST_ALIASES: dict[str, set[str]] = {
    "Aphex Twin": {
        "AFX", "Polygon Window", "Richard D. James", "Caustic Window",
        "The Tuss", "Blue Calx", "GAK", "Power-Pill", "Bradley Strider",
        "Soit-P.P.",
    },
    "Carl Craig": {
        "69", "Paperclip People", "Innerzone Orchestra",
        "Designer Music", "BFC", "Psyche",
    },
    "Derrick May": {
        "Mayday", "Rhythim Is Rhythim",
    },
    "Jeff Mills": {
        "The Wizard", "Purpose Maker", "Millsart",
    },
    "Underground Resistance": {
        "UR", "X-101", "X-102", "X-103", "Galaxy 2 Galaxy",
        "Los Hermanos", "World Power Alliance",
    },
    "Goldie": {
        "Metalheadz", "Rufige Kru",
    },
    "Juan Atkins": {
        "Model 500", "Cybotron", "Infiniti",
    },
    "Kevin Saunderson": {
        "Inner City", "E-Dancer", "Reese",
    },
    "Richie Hawtin": {
        "Plastikman", "F.U.S.E.", "Circuit Breaker",
        "Concept 1", "Chrome",
    },
    "The Prodigy": {
        "Liam Howlett",
    },
    "Orbital": {
        "Phil Hartnoll", "Paul Hartnoll",
    },
    "Autechre": {
        "ae", "Gescom",
    },
    "Squarepusher": {
        "Tom Jenkinson", "Chaos A.D.", "Duke of Harringay",
    },
    "Luke Vibert": {
        "Wagon Christ", "Plug", "Kerrier District", "Amen Andrews",
    },
    "Moodymann": {
        "Kenny Dixon Jr", "KDJ",
    },
    "Theo Parrish": {
        "Rotating Assembly",
    },
    "Laurent Garnier": {
        "Choice", "Lbs",
    },
    "Dave Clarke": {
        "Directional Force",
    },
    "Surgeon": {
        "British Murder Boys", "Force + Form",
    },
    "Regis": {
        "Sandwell District", "British Murder Boys",
    },
    "Andy C": {
        "Andy C & Shimon",
    },
    "Roni Size": {
        "Reprazent",
    },
    "DJ Shadow": {
        "Josh Davis",
    },
    "A Guy Called Gerald": {
        "Gerald Simpson",
    },
    "808 State": {
        "Graham Massey",
    },
    "LTJ Bukem": {
        "Good Looking Records",
    },
    "Burial": {
        "William Bevan",
    },
    "Four Tet": {
        "Kieran Hebden", "KH", "Percussions",
    },
    "Floating Points": {
        "Sam Shepherd",
    },
    "Skream": {
        "Oliver Jones",
    },
    "Benga": {
        "Beni Uthman",
    },
    "Photek": {
        "Rupert Parkes", "Studio Pressure",
    },
    "Ed Rush": {
        "Ed Rush & Optical",
    },
    "Shy FX": {
        "Andre Williams",
    },
    "Danny Byrd": set(),
    "Sasha": {
        "Alexander Coe",
    },
    "John Digweed": set(),
    "Paul Oakenfold": {
        "Perfecto",
    },
    "Carl Cox": set(),
    "Frankie Knuckles": {
        "The Godfather of House",
    },
    "Ron Hardy": set(),
    "Larry Heard": {
        "Mr Fingers", "Fingers Inc",
    },
    "Marshall Jefferson": set(),
    "DJ Pierre": {
        "Phuture",
    },
    "Lil Louis": {
        "Lil' Louis",
    },
    "Robert Hood": {
        "Floorplan", "Monobox",
    },
    "Mike Banks": {
        "Mad Mike",
    },
    "DJ Spooky": {
        "Paul Miller",
    },
    "Daft Punk": {
        "Thomas Bangalter", "Guy-Manuel de Homem-Christo",
    },
    "Fatboy Slim": {
        "Norman Cook", "Quentin Cook", "The Housemartins",
        "Beats International", "Pizzaman",
    },
    "Massive Attack": {
        "3D", "Daddy G", "Robert Del Naja", "Grant Marshall",
    },
    "Portishead": {
        "Geoff Barrow", "Beth Gibbons",
    },
    "The Chemical Brothers": {
        "Tom Rowlands", "Ed Simons", "Dust Brothers",
    },
    "Underworld": {
        "Karl Hyde", "Rick Smith", "Lemon Interupt",
    },
    "DJ Rashad": {
        "Rashad Harden",
    },
    "DJ Spinn": {
        "Spinn",
    },
    "RP Boo": {
        "Kavain Space",
    },
    "Ricardo Villalobos": {
        "Villalobos",
    },
    "Ben UFO": set(),
    "Pearson Sound": {
        "Ramadanman", "David Kennedy",
    },
    "Joy Orbison": {
        "Peter O'Grady",
    },
    "Objekt": {
        "TJ Hertz",
    },
    "Actress": {
        "Darren Cunningham",
    },
    "Kode9": {
        "Steve Goodman",
    },
    "Mala": {
        "Mark Lawrence", "Digital Mystikz",
    },
    "Coki": {
        "Digital Mystikz",
    },
    "Loefah": set(),
    "Pinch": {
        "Rob Ellis",
    },
    "Venetian Snares": {
        "Aaron Funk", "Last Step", "Speed Dealer Moms",
    },
    "Boards of Canada": {
        "BOC", "Mike Sandison", "Marcus Eoin",
    },
    "Kraftwerk": {
        "Ralf Huetter", "Florian Schneider",
    },
    "Derrick Carter": set(),
    "Green Velvet": {
        "Cajmere", "Curtis Jones",
    },
    "Kenny Larkin": {
        "Dark Comedy",
    },
    "Stacey Pullen": {
        "Silent Phase", "Kosmik Messenger",
    },
    "Eddie Fowlkes": set(),
    "Blake Baxter": set(),
    "Anthony Shake Shakir": {
        "Shake",
    },
    "Drexciya": {
        "James Stinson", "Gerald Donald", "Dopplereffekt",
        "Arpanet", "Heinrich Mueller",
    },
    "Basic Channel": {
        "Moritz von Oswald", "Mark Ernestus", "Maurizio",
        "Rhythm & Sound",
    },
    "Pan Sonic": {
        "Mika Vainio", "Ilpo Vaisanen",
    },
    "Helena Hauff": set(),
    "Nina Kraviz": set(),
    "Amelie Lens": set(),
    "Charlotte de Witte": {
        "Raving George",
    },
    "Peggy Gou": set(),
    "DJ Koze": {
        "Adolf Noise", "Monaco Schransen",
    },
    "Kerri Chandler": set(),
    "Todd Terry": {
        "Royal House", "Black Riot", "Swan Lake",
    },
    "Masters at Work": {
        "MAW", "Louie Vega", "Kenny Dope",
        "Nuyorican Soul",
    },
    "MK": {
        "Marc Kinchen",
    },
    "Armand Van Helden": set(),
    "David Morales": {
        "The Boss",
    },
    "Danny Tenaglia": set(),
    "Frankie Bones": set(),
    "Lenny Dee": set(),
    "Marc Acardipane": {
        "The Mover", "Pilldriver", "Marshall Masters",
    },
    "Noisia": {
        "Nik Roos", "Martijn van Sonderen", "Thijs de Vlieger",
        "Vision", "Outer Edges",
    },
    "DJ Hype": set(),
    "Fabio": {
        "Fabio & Grooverider",
    },
    "Grooverider": {
        "Fabio & Grooverider",
    },
    "Bryan Gee": set(),
    "Calibre": {
        "Dominick Martin",
    },
    "High Contrast": {
        "Lincoln Barrett",
    },
}

# ── Build inverted index at module load time ────────────────────────────
# Maps every alias (lowercased) -> canonical name, so reverse lookups are O(1).
_ALIAS_TO_CANONICAL: dict[str, str] = {}

for _canonical, _aliases in ARTIST_ALIASES.items():
    _canonical_lower = _canonical.lower()
    # Map the canonical name to itself (so lookups for canonical names work too).
    _ALIAS_TO_CANONICAL[_canonical_lower] = _canonical
    for _alias in _aliases:
        _alias_lower = _alias.lower()
        # If an alias appears under multiple canonicals (e.g. "British Murder Boys"
        # under both Surgeon and Regis), the last write wins.  This is acceptable
        # because expand_aliases will still find the artist via either canonical.
        _ALIAS_TO_CANONICAL[_alias_lower] = _canonical

# Clean up module-level loop variables to avoid polluting the namespace.
del _canonical, _aliases, _canonical_lower
# _alias and _alias_lower may not exist if the last artist had an empty alias set.
try:
    del _alias, _alias_lower
except NameError:
    pass


def expand_aliases(name: str) -> set[str]:
    """Return all known names for the given artist (canonical + all aliases).

    Performs a case-insensitive lookup.  If the input is a known alias, the
    canonical name and all sibling aliases are returned.  If the input is a
    canonical name, the canonical name plus all its aliases are returned.

    Args:
        name: An artist name or alias to expand.

    Returns:
        A set containing the canonical name and all known aliases.  If the
        name is not recognized, returns a set containing only the original
        name.
    """
    canonical = _ALIAS_TO_CANONICAL.get(name.lower().strip())
    if canonical is None:
        return {name}
    # Return canonical name + all aliases.
    return {canonical} | ARTIST_ALIASES[canonical]


def get_canonical_name(name_or_alias: str) -> str | None:
    """If the input is a known alias, return the canonical name.

    Case-insensitive lookup.

    Args:
        name_or_alias: An artist name or alias.

    Returns:
        The canonical artist name, or None if not recognized.
    """
    return _ALIAS_TO_CANONICAL.get(name_or_alias.lower().strip())


def get_all_artist_names() -> set[str]:
    """Return a flat set of all known artist names (canonicals + aliases).

    Useful for building search caches or prefix-match indexes.

    Returns:
        A set of all canonical names and aliases.
    """
    all_names: set[str] = set()
    for canonical, aliases in ARTIST_ALIASES.items():
        all_names.add(canonical)
        all_names.update(aliases)
    return all_names


# ═════════════════════════════════════════════════════════════════════════
# 5. GENRE EXTRACTION HELPER
# ═════════════════════════════════════════════════════════════════════════
# Extracts genre keywords from a query by matching against all known genre
# names in GENRE_ADJACENCY.  Longest-match-first prevents "house" from
# matching inside "acid house".

# Pre-sorted genre names: longest first, for greedy matching.
_GENRES_BY_LENGTH: list[str] = sorted(GENRE_ADJACENCY.keys(), key=len, reverse=True)


def extract_query_genres(query: str) -> set[str]:
    """Extract genre keywords from a query by matching against known genre names.

    Uses longest-match-first strategy so that e.g. "acid house" is matched
    as a single genre before "house" can match on its own.

    Args:
        query: The user's search query text.

    Returns:
        A set of matched genre name strings (lowercase).
    """
    query_lower = query.lower()
    matched: set[str] = set()
    # Track character positions already consumed by a longer match.
    consumed: set[int] = set()

    for genre in _GENRES_BY_LENGTH:
        start = 0
        while True:
            idx = query_lower.find(genre, start)
            if idx == -1:
                break
            end = idx + len(genre)
            # Check that none of the positions are already consumed.
            positions = set(range(idx, end))
            if not positions & consumed:
                # Verify word boundaries: the character before the match
                # (if any) and after the match (if any) should not be
                # alphanumeric, to avoid matching inside unrelated words.
                before_ok = idx == 0 or not query_lower[idx - 1].isalnum()
                after_ok = end == len(query_lower) or not query_lower[end].isalnum()
                if before_ok and after_ok:
                    matched.add(genre)
                    consumed.update(positions)
            start = idx + 1

    return matched
