"""Phrase dictionaries used by the deterministic rules.

Guidelines for contributors:
  * Phrases are matched case-insensitively on word boundaries.
  * ``'`` matches straight and curly apostrophes; a space matches any whitespace.
  * ``*`` stands for exactly one word; ``(a|b)`` alternation is allowed.
  * Single words are never flagged on their own by L01 -- only clusters are.
  * Each entry should be a pattern that is *characteristic*, not merely common.
"""

from __future__ import annotations

import re

_WORD = r"[\w'’\-]+"


def _phrase_to_regex(p: str) -> str:
    out = []
    i = 0
    while i < len(p):
        c = p[i]
        if c == "\\" and i + 1 < len(p):
            i += 1
            out.append(re.escape(p[i]))
        elif c == "*":
            out.append(_WORD)
        elif c == "'":
            out.append("['’]")
        elif c == " ":
            out.append(r"\s+")
        elif c in "(|)?":
            out.append(c)
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def compile_phrases(phrases: list[str] | tuple[str, ...]) -> re.Pattern[str]:
    alts = sorted({_phrase_to_regex(p) for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<![\w'’])(?:" + "|".join(alts) + r")(?![\w'’])", re.I)


# --- L01: vocabulary strongly over-represented in LLM output ----------------
AI_VOCAB = (
    "delve", "delves", "delving", "tapestry", "testament", "realm", "realms", "embark", "embarking",
    "beacon", "multifaceted", "intricate", "intricacies", "pivotal", "paramount", "vibrant", "bustling",
    "seamless", "seamlessly", "leverage", "leveraging", "harness", "harnessing", "unlock", "unlocking",
    "unleash", "unleashing", "elevate", "elevating", "empower", "empowering", "foster", "fostering",
    "underscore", "underscores", "underscoring", "showcase", "showcasing", "holistic", "synergy",
    "synergies", "transformative", "meticulous", "meticulously", "commendable", "notably", "moreover",
    "furthermore", "navigate", "navigating", "landscape", "ever-evolving", "ever-changing", "cutting-edge",
    "game-changer", "game-changing", "groundbreaking", "revolutionize", "revolutionizing", "streamline",
    "streamlining", "robust", "nuanced", "crucial", "vital", "comprehensive", "invaluable", "endeavor",
    "endeavors", "bolster", "bolstering", "resonate", "resonates", "spearhead", "captivating",
    "enigmatic", "profound", "interplay", "plethora", "myriad", "boast", "boasts", "nestled",
    "unparalleled", "unwavering", "indelible", "labyrinth", "symphony", "kaleidoscope", "treasure trove",
    "dynamic", "innovative", "ecosystem", "paradigm", "journey", "elevated", "curated", "bespoke",
    "harmonious", "align", "aligns", "intrinsically", "inherently", "facilitate", "facilitates",
    "optimize", "optimizing", "utilize", "utilizing", "crucially", "importantly", "additionally",
)

# --- L02: vague marketing adjectives used in stacks ----------------------------
BUZZ_ADJECTIVES = (
    "innovative", "scalable", "seamless", "robust", "cutting-edge", "next-generation", "next-gen",
    "world-class", "best-in-class", "holistic", "dynamic", "agile", "synergistic", "transformative",
    "end-to-end", "state-of-the-art", "mission-critical", "user-centric", "customer-centric",
    "data-driven", "future-proof", "streamlined", "powerful", "comprehensive", "intuitive", "efficient",
    "flexible", "reliable", "secure", "modern", "smart", "intelligent", "personalized", "actionable",
    "impactful", "game-changing", "groundbreaking", "revolutionary", "unparalleled", "unrivaled",
    "industry-leading", "enterprise-grade", "frictionless", "effortless", "elegant", "sustainable",
    "versatile", "adaptive", "proactive", "strategic", "optimized", "integrated", "immersive",
    "engaging", "vibrant", "bespoke", "curated", "premium", "exceptional", "unique",
)

# --- L03: generic scene-setting introductions ---------------------------------
SCENE_SETTING = (
    "in today's * (world|landscape|age|era|environment|economy|market|society)",
    "in today's * * (world|landscape|age|era|environment|economy|market|society)",
    "in today's * * * (world|landscape|age|era|environment|economy|market|society)",
    "in today's (world|landscape|age|era)",
    "in an (era|age|world) (of|where|defined by)",
    "in the (ever-evolving|ever-changing|rapidly evolving|fast-paced|dynamic|modern|digital) (world|landscape|realm|field|era) of",
    "in the (ever-evolving|ever-changing|rapidly evolving|fast-paced|dynamic) (world|landscape|realm)",
    "as (technology|ai|the world|society|the industry|the digital landscape) continues to (evolve|advance|grow|change)",
    "now more than ever",
    "imagine a world where",
    "picture this",
    "in recent years, * has (become|emerged|gained)",
    "in this (article|post|guide|blog post), (we|i)('ll| will) (explore|discuss|dive|delve|look)",
    "whether you're a (seasoned|beginner|novice|veteran)",
    "have you ever wondered",
    "it's no secret that",
    "it is no secret that",
    "since the dawn of",
    "throughout history",
)

# --- L04: empty intensifiers ---------------------------------------------------
INTENSIFIERS = (
    "truly", "fundamentally", "incredibly", "absolutely", "genuinely", "deeply", "profoundly",
    "remarkably", "undeniably", "unquestionably", "extremely", "literally", "utterly", "totally",
    "exceptionally", "tremendously", "immensely", "vastly", "wholly", "entirely", "significantly",
    "drastically", "dramatically", "radically", "inherently", "intrinsically", "essentially", "simply",
    "really", "very", "super",
)

# --- L05: stock transitional phrases ------------------------------------------
STOCK_TRANSITIONS = (
    "here's the thing", "that said", "that being said", "with that said", "it's worth noting",
    "it is worth noting", "it's important to note", "it is important to note", "it's important to remember",
    "it is important to remember", "it's worth mentioning", "at the end of the day", "when it comes to",
    "in conclusion", "in summary", "to sum up", "to summarize", "in essence", "to put it simply",
    "simply put", "put simply", "let's dive in", "let's dive into", "let's dive deeper", "let's break it down",
    "let's break this down", "let's unpack", "let's explore", "without further ado", "it goes without saying",
    "needless to say", "first and foremost", "last but not least", "all in all", "in a nutshell",
    "the bottom line is", "bottom line:", "the key takeaway", "key takeaways", "in other words",
    "with this in mind", "having said that", "on the other hand", "by the same token",
)
SENTENCE_INITIAL_TRANSITIONS = (
    "moreover", "furthermore", "additionally", "notably", "importantly", "ultimately", "consequently",
    "crucially", "interestingly", "indeed", "overall",
)

# --- L06: decorative / cliché metaphors -------------------------------------
CLICHE_METAPHORS = (
    "double-edged sword", "tip of the iceberg", "game changer", "game-changer", "paradigm shift",
    "north star", "secret sauce", "low-hanging fruit", "move the needle", "moving the needle",
    "a symphony of", "tapestry of", "rich tapestry", "beacon of", "testament to", "at the forefront of",
    "pave the way", "paves the way", "paving the way", "the heart of", "at its core", "unlock the potential",
    "unlock the power", "unleash the power", "unleash the potential", "harness the power",
    "navigate the (complex|ever-changing|evolving|intricate|complexities)", "navigate the * landscape",
    "navigating the (complex|ever-changing|evolving|intricate|complexities)", "treasure trove",
    "the fabric of", "stands as a", "serves as a testament", "a new frontier", "the next frontier",
    "seismic shift", "the building blocks of", "the cornerstone of", "a cornerstone of", "dance of",
    "delicate dance", "delicate balance", "in the realm of", "the world of", "a world of",
    "on this journey", "your * journey", "a journey of", "the journey of", "the landscape of",
    "shaping the future", "shape the future", "the future of *", "a breath of fresh air",
    "uncharted (territory|waters)", "push the boundaries", "pushing the boundaries", "raise the bar",
    "raising the bar", "level up", "next level", "take it to the next level", "a perfect storm",
    "the elephant in the room", "silver bullet", "a force to be reckoned with", "lifeblood of",
    "the tip of the spear", "a goldmine", "gold standard", "bridge the gap", "bridging the gap",
    "the missing piece", "the puzzle", "deep dive", "deep-dive", "ever-shifting sands", "ripple effect",
)

# --- R04: manufactured revelation -------------------------------------------
REVELATION = (
    "here's what most people (miss|don't realize|get wrong|don't know|overlook)",
    "here's the (thing|kicker|catch|twist|truth|secret|deal|problem|reality)",
    "here's where it gets interesting", "here's why (this|it|that) matters", "here's what (nobody|no one) tells you",
    "what (nobody|no one) (tells you|talks about)", "the (uncomfortable|hard|brutal|simple|real|honest|dirty) truth",
    "the truth is", "the reality is", "the secret is", "the secret to", "the real (reason|story|question|problem|answer|magic|opportunity|issue|lesson|power|value|win|cost|risk)",
    "what most people don't (realize|know|understand)", "most people don't realize", "let that sink in",
    "plot twist", "spoiler:", "spoiler alert", "but wait", "the kicker\\?", "the twist\\?", "and here's the kicker",
    "it turns out", "turns out", "this is where * comes in", "that's where * comes in", "enter *:",
)

# --- R05: dramatic one-sentence conclusions --------------------------------
DRAMATIC_CLOSERS = (
    "and that changes everything", "that changes everything", "this changes everything",
    "and that's the point", "that's the point", "that's the whole point", "and that's what matters",
    "that's what matters", "that's what truly matters", "and that's what truly matters",
    "that's the power of *", "that's the beauty of *", "and that's the magic", "that's the magic",
    "full stop", "period\\.", "end of story", "it's that simple", "it really is that simple",
    "and that makes all the difference", "that makes all the difference", "and it shows",
    "the future is (here|now|bright)", "the rest is history", "and honestly\\? that's enough",
    "that's it\\.", "that's the post", "think about that", "let that sink in", "and that's okay",
    "and that's everything", "and it matters", "this matters", "it matters",
)

# --- C03 / C05: sweeping claims & manufactured significance ------------------
SWEEPING = (
    "changes everything", "change everything", "revolutionize *", "revolutionizing *", "will transform",
    "will redefine", "will reshape", "will forever change", "forever changed", "unprecedented",
    "limitless", "endless possibilities", "the possibilities are endless", "like never before",
    "in ways we never imagined", "in ways we can't imagine", "everyone knows", "it's clear that",
    "it is clear that", "there's no doubt", "there is no doubt", "without a doubt", "undoubtedly",
    "the future of (work|business|technology|ai|everything|humanity)", "the only way", "the key to success",
    "nothing will ever be the same", "never be the same", "(a|this|the) new era", "a new age", "the dawn of",
    "for everyone", "every industry", "every aspect of", "all aspects of",
)
SIGNIFICANCE = (
    "this is huge", "game-changing", "game changer", "game-changer", "groundbreaking", "revolutionary",
    "monumental", "a pivotal moment", "pivotal moment", "marks a turning point", "a turning point",
    "a watershed moment", "historic moment", "a defining moment", "landmark", "paradigm-shifting",
    "earth-shattering", "mind-blowing", "seismic", "reshaping", "redefining", "reimagining",
    "reshapes", "redefines", "reimagines", "fundamentally reshaping", "fundamentally changing",
    "fundamentally transforming",
)

# --- C04: vague referents ------------------------------------------------------
VAGUE_REFERENTS = (
    "things", "stuff", "something", "aspects", "factors", "elements", "solutions", "various", "numerous",
    "a variety of", "a wide range of", "a range of", "a number of", "certain", "many people",
    "some people", "people", "everyone", "everything", "someone", "in many ways", "in some ways",
    "on many levels", "and more", "and so on", "and so forth", "etc",
)

# --- C01/C06: abstract nouns (low specificity signal) --------------------------
ABSTRACT_NOUNS = frozenset("""
opportunity opportunities solution solutions innovation innovations growth potential value values impact
journey landscape transformation success excellence possibility possibilities experience experiences
approach approaches strategy strategies ecosystem ecosystems future efficiency efficiencies productivity
synergy synergies paradigm paradigms framework frameworks insight insights engagement outcomes outcome
capabilities capability performance quality process processes results result benefit benefits challenge
challenges needs goals goal vision mission purpose meaning connection connections community communities
progress change changes way ways world society humanity technology technologies resilience agility
collaboration communication alignment empowerment wellbeing well-being balance harmony creativity
excellence leadership mindset culture cultures dynamics stakeholders stakeholder environment
environments space spaces level levels area areas aspect aspects element elements factor factors
""".split())

# --- C07: generic recommendations -------------------------------------------
GENERIC_ADVICE = (
    "stay (curious|informed|up to date|up-to-date|ahead of the curve|consistent|focused|positive|flexible|open-minded|agile)",
    "embrace (change|the change|innovation|the future|new technologies|the journey|uncertainty|failure|the process)",
    "focus on what (matters|truly matters|really matters)", "start small", "be consistent", "be patient",
    "be authentic", "be yourself", "consider your (needs|goals|options|specific needs|unique needs)",
    "it's essential to", "it is essential to", "it's crucial to", "it is crucial to", "it's important to",
    "it is important to", "make sure to", "remember to", "don't be afraid to", "keep in mind that",
    "communicate (clearly|openly|effectively|regularly)", "foster a culture of", "invest in (yourself|your team|quality|the right tools)",
    "prioritize (what matters|your goals|self-care|quality|your well-being)", "set clear goals",
    "find what works for you", "listen to your (customers|users|audience|team)", "know your audience",
    "continuously (learn|improve|iterate|adapt)", "never stop learning", "adapt to change",
    "leverage the power of", "take a holistic approach", "seek feedback", "think outside the box",
    "work smarter, not harder", "the key is to", "the key is", "best practices",
)

# --- D04: verbose phrases with tighter alternatives -----------------------------
WORDY = {
    "in order to": "to",
    "due to the fact that": "because",
    "owing to the fact that": "because",
    "in light of the fact that": "because",
    "despite the fact that": "although",
    "at this point in time": "now",
    "at the present time": "now",
    "in the event that": "if",
    "for the purpose of": "to / for",
    "with regard to": "about",
    "with respect to": "about",
    "in terms of": "(rephrase)",
    "it should be noted that": "(cut)",
    "it is important to note that": "(cut)",
    "it's important to note that": "(cut)",
    "the fact that": "(rephrase)",
    "a large number of": "many",
    "a significant number of": "many",
    "the vast majority of": "most",
    "has the ability to": "can",
    "have the ability to": "can",
    "is able to": "can",
    "are able to": "can",
    "each and every": "each / every",
    "first and foremost": "first",
    "absolutely essential": "essential",
    "end result": "result",
    "past history": "history",
    "future plans": "plans",
    "plays a (crucial|key|vital|significant|pivotal|critical|important) role in": "(say what it does)",
    "serves as a": "is a",
    "in the process of": "(cut)",
    "on a daily basis": "daily",
    "on a regular basis": "regularly",
    "at the end of the day": "(cut)",
    "it is worth noting that": "(cut)",
    "it's worth noting that": "(cut)",
    "when it comes to": "for / with",
    "in today's day and age": "today",
    "make a decision": "decide",
    "take into consideration": "consider",
    "come to the conclusion": "conclude",
    "a wide variety of": "many",
    "in close proximity to": "near",
    "prior to": "before",
    "subsequent to": "after",
    "in spite of the fact that": "although",
    "whether or not": "whether",
    "the reason why is because": "because",
}

# --- T01: unsolicited validation ---------------------------------------------
VALIDATION = (
    "great question", "what a great question", "excellent question", "good question", "fantastic question",
    "that's a great question", "that's a fantastic", "that's an excellent", "you're absolutely right",
    "you are absolutely right", "you're right to", "excellent point", "great point", "what a great",
    "i love this question", "i love that you", "love this", "brilliant question", "astute observation",
    "you've hit on", "you raise an excellent point", "absolutely!", "certainly!", "of course!", "great idea",
)

# --- T02: false profundity ---------------------------------------------------------
PROFUNDITY = (
    "it's not about * it's about", "is a journey, not a destination", "a journey, not a destination",
    "the real * was * all along", "and perhaps that's the (most important|greatest|real) lesson",
    "perhaps that's the point", "maybe that's the point", "because at the end of the day",
    "in the end, it's", "in the end, what matters", "the only limit is", "the future belongs to",
    "the future is (bright|ours|now|here)", "what truly matters is", "what really matters is",
    "the most important thing is", "that is the true", "that's the true", "remind(s)? us that",
    "a reminder that", "teaches us that", "at its heart,", "at its core,", "in a world where",
    "after all, isn't that", "isn't that what * is all about", "and isn't that", "the beauty of it is",
    "true (success|growth|innovation|leadership|happiness|value) (lies|comes|is)",
)

# --- T03: artificial emotional intensity --------------------------------------------
EMOTIVE = (
    "thrilled", "so excited", "excited to (share|announce|introduce)", "absolutely love", "absolutely amazing",
    "amazing", "incredible", "mind-blowing", "mind blowing", "insane", "insanely", "jaw-dropping",
    "game-changing", "blown away", "obsessed", "can't wait", "buckle up", "hold on tight", "brace yourself",
    "wild", "epic", "unbelievable", "stunning", "breathtaking", "life-changing", "next-level",
)

# --- T04: performative honesty ----------------------------------------------------
HONESTY = (
    "i'll be honest", "i will be honest", "to be honest", "to be completely honest", "honestly,",
    "let's be real", "let's be honest", "let me be clear", "let me be honest", "no sugarcoating",
    "no sugar-coating", "i won't sugarcoat", "the brutal truth", "brutally honest", "i'm not going to lie",
    "not gonna lie", "real talk", "truth be told", "frankly,", "candidly,", "i'll be straight with you",
    "let me be straight", "here's my honest take", "my honest take",
)

# --- T05: hedging & false balance ---------------------------------------------------
HEDGES = (
    "may", "might", "could", "possibly", "potentially", "perhaps", "arguably", "somewhat", "relatively",
    "generally", "typically", "often", "usually", "tends to", "tend to", "to some extent", "to a certain extent",
    "in some cases", "in many cases", "it depends", "it's possible that", "it is possible that", "seems to",
    "appears to", "likely", "probably", "more or less", "kind of", "sort of",
)
FALSE_BALANCE = (
    "there's no one-size-fits-all", "there is no one-size-fits-all", "no one-size-fits-all",
    "it depends on your (specific|unique|individual|particular) (needs|goals|situation|circumstances|use case)",
    "ultimately, the (best|right) (choice|option|approach|decision) depends", "ultimately, it depends",
    "both have their (pros and cons|merits|strengths)", "each has its (pros and cons|merits|strengths)",
    "there are pros and cons", "there are valid arguments on both sides", "both sides have valid points",
    "it's a matter of personal preference", "it really depends", "the answer is: it depends",
    "the choice is yours", "what works for one * may not work for another",
)

# --- T06: inappropriate conversational familiarity -----------------------------
FAMILIARITY = (
    "let's dive in", "let's dive right in", "let's get started", "let's jump in", "buckle up",
    "don't worry", "trust me", "you've got this", "you got this", "i hope this helps", "hope this helps",
    "happy coding", "happy writing", "happy learning", "feel free to", "let me know if", "hey there",
    "spoiler alert", "fear not", "rest assured", "no worries", "pro tip", "pro-tip", "here's a fun fact",
    "fun fact", "you're not alone", "we've all been there", "sound familiar\\?", "ready\\? let's go",
    "i'd be happy to", "i'm happy to help", "happy to help", "i'm here to help", "let's walk through",
    "grab a coffee", "stick around",
)
