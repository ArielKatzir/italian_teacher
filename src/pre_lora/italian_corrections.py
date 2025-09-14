"""
Comprehensive Italian Error Correction Database

This module contains extensive Italian error patterns and corrections
that can be used for both pre-LoRA correction and as training data
for future LoRA fine-tuning.
"""

import re
from typing import Dict, List, Pattern, Tuple

from ..core.error_tolerance import ErrorType


class ItalianCorrectionsDB:
    """
    Comprehensive database of Italian error patterns and corrections.

    This database contains hundreds of common Italian learner errors
    organized by error type, with contextual variations and explanations
    suitable for generating LoRA training data.
    """

    def __init__(self):
        self.corrections = self._initialize_corrections()
        self.error_patterns = self._initialize_error_patterns()

    def _initialize_corrections(self) -> Dict[str, Dict[str, str]]:
        """Initialize comprehensive correction mappings by category."""
        return {
            # Verb conjugation errors
            "verb_essere": {
                "io essere": "io sono",
                "tu essere": "tu sei",
                "lui essere": "lui è",
                "lei essere": "lei è",
                "noi essere": "noi siamo",
                "voi essere": "voi siete",
                "loro essere": "loro sono",
                "io sono essere": "io sono",
                "ho essere": "sono",
                "hai essere": "sei",
                "ha essere": "è",
            },
            "verb_avere": {
                "io avere": "io ho",
                "tu avere": "tu hai",
                "lui avere": "lui ha",
                "lei avere": "lei ha",
                "noi avere": "noi abbiamo",
                "voi avere": "voi avete",
                "loro avere": "loro hanno",
                "sono avere": "ho",
                "sei avere": "hai",
                "è avere": "ha",
            },
            "verb_andare": {
                "io andare": "io vado",
                "tu andare": "tu vai",
                "lui andare": "lui va",
                "io ando": "io vado",
                "tu andi": "tu vai",
                "lui anda": "lui va",
                "io vendo": "io vado",  # common confusion
                "io va": "io vado",
                "tu va": "tu vai",
            },
            "verb_fare": {
                "io fare": "io faccio",
                "tu fare": "tu fai",
                "lui fare": "lui fa",
                "io faco": "io faccio",
                "io fo": "io faccio",
                "tu faci": "tu fai",
                "lui face": "lui fa",
            },
            "verb_stare": {
                "io stare": "io sto",
                "tu stare": "tu stai",
                "lui stare": "lui sta",
                "io sto essere": "io sto",
                "io stando": "io sto",
                "tu stando": "tu stai",
            },
            # Gender agreement errors
            "gender_articles": {
                "un casa": "una casa",
                "un macchina": "una macchina",
                "un ragazza": "una ragazza",
                "un pizza": "una pizza",
                "un birra": "una birra",
                "un strada": "una strada",
                "un banca": "una banca",
                "un scuola": "una scuola",
                "una uomo": "un uomo",
                "una ragazzo": "un ragazzo",
                "una libro": "un libro",
                "una telefono": "un telefono",
                "una computer": "un computer",
                "la problema": "il problema",
                "la sistema": "il sistema",
                "la programma": "il programma",
                "la teorema": "il teorema",
                "il persona": "la persona",
                "il mano": "la mano",
                "il foto": "la foto",
                "il radio": "la radio",
            },
            "gender_adjectives": {
                "ragazza alto": "ragazza alta",
                "casa piccolo": "casa piccola",
                "macchina rosso": "macchina rossa",
                "pizza buono": "pizza buona",
                "uomo bella": "uomo bello",
                "ragazzo carina": "ragazzo carino",
                "libro interessanta": "libro interessante",
                "problema difficila": "problema difficile",
            },
            # Preposition errors
            "prepositions_place": {
                "vado in casa": "vado a casa",
                "sono in casa": "sono a casa",
                "vado in Roma": "vado a Roma",
                "sono in Italia": "sono in Italia",  # correct
                "vado in Italia": "vado in Italia",  # correct
                "vengo da Italia": "vengo dall'Italia",
                "vengo da Roma": "vengo da Roma",  # correct
                "arrivo da Milano": "arrivo da Milano",  # correct
                "vado nella scuola": "vado a scuola",
                "sono nella scuola": "sono a scuola",
                "vado nel lavoro": "vado al lavoro",
                "sono nel lavoro": "sono al lavoro",
            },
            "prepositions_time": {
                "nella mattina": "di mattina",
                "nella sera": "di sera",
                "nel pomeriggio": "di pomeriggio",
                "in lunedì": "di lunedì",
                "in martedì": "di martedì",
                "a mattina": "di mattina",
                "a sera": "di sera",
            },
            "prepositions_verbs": {
                "penso a fare": "penso di fare",
                "cerco per trovare": "cerco di trovare",
                "smetto per fumare": "smetto di fumare",
                "comincio per studiare": "comincio a studiare",
                "finisco per lavorare": "finisco di lavorare",
                "aiuto per fare": "aiuto a fare",
            },
            # False friends and vocabulary
            "false_friends": {
                "sono eccitato": "sono entusiasta",  # eccitato = sexually excited
                "mia moglie è incinta": "mia moglie aspetta un bambino",  # more polite
                "sono confuso": "sono confuso",  # can be correct but often "sono perplesso"
                "realizzo che": "mi rendo conto che",  # realize ≠ realizzare
                "attualmente": "in realtà",  # attualmente = currently
                "eventualmente": "forse",  # eventualmente = eventually
                "preservativo": "profilattico",  # preservativo = condom
            },
            # Word order errors
            "word_order": {
                "molto bene è": "è molto bene",
                "troppo difficile è": "è troppo difficile",
                "sempre io vado": "io vado sempre",
                "già io ho fatto": "ho già fatto",
                "mai io non vado": "non vado mai",
                "spesso noi mangiamo": "noi mangiamo spesso",
                "ancora tu non capisci": "tu non capisci ancora",
            },
            # Double negatives (correct in Italian but confusing for learners)
            "negation": {
                "non ho niente": "non ho niente",  # correct - just for reference
                "non vedo nessuno": "non vedo nessuno",  # correct
                "non vado mai": "non vado mai",  # correct
                "non ho mai": "non ho mai",  # correct but incomplete
                "io non non vado": "io non vado",  # double negative error
                "non ho no tempo": "non ho tempo",
            },
            # More irregular verbs
            "verb_venire": {
                "io venire": "io vengo",
                "tu venire": "tu vieni",
                "lui venire": "lui viene",
                "io veno": "io vengo",
                "tu veni": "tu vieni",
                "lui vene": "lui viene",
                "io venga": "io vengo",  # subjunctive confusion
            },
            "verb_volere": {
                "io volere": "io voglio",
                "tu volere": "tu vuoi",
                "lui volere": "lui vuole",
                "io volio": "io voglio",
                "tu voli": "tu vuoi",
                "lui vole": "lui vuole",
                "io volo": "io voglio",  # common confusion with volare
            },
            "verb_potere": {
                "io potere": "io posso",
                "tu potere": "tu puoi",
                "lui potere": "lui può",
                "io poto": "io posso",
                "tu poti": "tu puoi",
                "lui pote": "lui può",
                "io potro": "io potrò",  # future tense error
            },
            "verb_dovere": {
                "io dovere": "io devo",
                "tu dovere": "tu devi",
                "lui dovere": "lui deve",
                "io dovo": "io devo",
                "tu dovi": "tu devi",
                "lui dove": "lui deve",
            },
            "verb_sapere": {
                "io sapere": "io so",
                "tu sapere": "tu sai",
                "lui sapere": "lui sa",
                "io sapo": "io so",
                "tu sapi": "tu sai",
                "lui sape": "lui sa",
                "io conosco Roma": "io conosco Roma",  # correct - different meaning
            },
            "verb_bere": {
                "io bere": "io bevo",
                "tu bere": "tu bevi",
                "lui bere": "lui beve",
                "io bero": "io bevo",
                "tu beri": "tu bevi",
                "lui bere": "lui beve",
            },
            "verb_dare": {
                "io dare": "io do",
                "tu dare": "tu dai",
                "lui dare": "lui dà",
                "io dao": "io do",
                "tu dari": "tu dai",
                "lui dae": "lui dà",
                "lui da": "lui dà",  # missing accent
            },
            # Past tense errors
            "passato_prossimo": {
                "io sono andato": "io sono andato",  # correct
                "io ho andato": "io sono andato",  # movement verbs use essere
                "io sono mangiato": "io ho mangiato",  # non-movement verbs use avere
                "io ho arrivato": "io sono arrivato",
                "io sono bevuto": "io ho bevuto",
                "io ho venuto": "io sono venuto",
                "io sono comprato": "io ho comprato",
                "io ho partito": "io sono partito",
                "io sono letto": "io ho letto",
                "io ho nato": "io sono nato",
            },
            "past_participles": {
                "ho mangiat": "ho mangiato",
                "ho beut": "ho bevuto",
                "ho pres": "ho preso",
                "ho venud": "sono venuto",
                "ho andat": "sono andato",
                "ho vist": "ho visto",
                "ho facet": "ho fatto",
                "ho dit": "ho detto",
                "ho scritt": "ho scritto",
            },
            # Pronoun errors
            "direct_pronouns": {
                "io vedo lui": "io lo vedo",
                "io vedo lei": "io la vedo",
                "io compro esso": "io lo compro",
                "io mangio essa": "io la mangio",
                "vedo te": "ti vedo",
                "chiamo voi": "vi chiamo",
                "amo tu": "ti amo",
                "conosco loro": "li conosco",
            },
            "indirect_pronouns": {
                "do a lui": "gli do",
                "do a lei": "le do",
                "parlo a te": "ti parlo",
                "scrivo a voi": "vi scrivo",
                "telefono a loro": "gli telefono",
                "regalo a lui": "gli regalo",
                "spiego a lei": "le spiego",
            },
            "reflexive_verbs": {
                "io chiamo Marco": "io chiamo Marco",  # correct - not reflexive
                "io chiamo me": "io mi chiamo",  # reflexive
                "io lavo me": "io mi lavo",
                "tu vesti te": "tu ti vesti",
                "lui sveglia se": "lui si sveglia",
                "noi diveriamo": "noi ci divertiamo",
                "voi preparate voi": "voi vi preparate",
                "loro alzano se": "loro si alzano",
            },
            # More preposition combinations
            "complex_prepositions": {
                "vado da medico": "vado dal medico",
                "vengo da farmacia": "vengo dalla farmacia",
                "passo da ufficio": "passo dall'ufficio",
                "studio da professore": "studio dal professore",
                "lavoro da tre anni": "lavoro da tre anni",  # correct - duration
                "studio da due ore": "studio da due ore",  # correct - duration
                "vado a medico": "vado dal medico",  # wrong preposition
                "sono da Milano": "sono di Milano",  # origin vs location
                "vado in Venezia": "vado a Venezia",  # cities use 'a'
                "sono in Bologna": "sono a Bologna",
                "vivo a Italia": "vivo in Italia",  # countries use 'in'
                "abito a Francia": "abito in Francia",
            },
            "prepositions_transport": {
                "vado con macchina": "vado in macchina",
                "viaggio con treno": "viaggio in treno",
                "arrivo con aereo": "arrivo in aereo",
                "vado con piedi": "vado a piedi",
                "viaggio con bicicletta": "vado in bicicletta",
                "arrivo con autobus": "arrivo in autobus",
                "vado in piedi": "vado a piedi",
            },
            # Subjunctive errors (common advanced mistakes)
            "subjunctive_triggers": {
                "penso che lui è": "penso che lui sia",
                "credo che tu hai": "credo che tu abbia",
                "spero che loro vanno": "spero che loro vadano",
                "voglio che voi fate": "voglio che voi facciate",
                "è importante che tu studi": "è importante che tu studi",  # correct
                "bisogna che io vado": "bisogna che io vada",
                "può darsi che lui viene": "può darsi che lui venga",
                "ho paura che non funziona": "ho paura che non funzioni",
            },
            # Conditional mood errors
            "conditional": {
                "io volerei": "io vorrei",  # wrong verb
                "tu potesti": "tu potresti",
                "lui doverebbe": "lui dovrebbe",  # correct
                "noi faremo": "noi faremmo",  # future vs conditional
                "voi venite": "voi verreste",
                "loro sarebbero": "loro sarebbero",  # correct
                "se io sarei": "se io fossi",  # conditional in 'if' clause
                "se tu avresti": "se tu avessi",
            },
            # More false friends and common mistakes
            "advanced_false_friends": {
                "sono embarrassato": "sono imbarazzato",
                "è molto sensitivo": "è molto sensibile",
                "questo è delizioso": "questo è delizioso",  # correct
                "lui è grosso": "lui è grande",  # grosso = thick/coarse
                "una libreria": "una libreria",  # correct = bookstore
                "una biblioteca": "una biblioteca",  # correct = library
                "ufficio": "ufficio",  # correct
                "fabbrica": "fabbrica",  # correct = factory
                "magazzino": "magazzino",  # correct = warehouse
                "attendo per te": "ti aspetto",  # attendere = wait formally
                "frequento università": "frequento l'università",
                "argomento": "argomento",  # correct = topic
                "discussione": "discussione",  # correct = discussion
                "confronto": "confronto",  # correct = comparison
            },
            # Articles with prepositions
            "articulated_prepositions": {
                "vado a il cinema": "vado al cinema",
                "vengo da il lavoro": "vengo dal lavoro",
                "passo per il centro": "passo per il centro",  # correct
                "studio in la biblioteca": "studio nella biblioteca",
                "mangio in il ristorante": "mangio nel ristorante",
                "compro da la farmacia": "compro dalla farmacia",
                "arrivo da la stazione": "arrivo dalla stazione",
                "vado su la montagna": "vado sulla montagna",
                "gioco su il campo": "gioco sul campo",
            },
            # Time expressions
            "time_expressions": {
                "alle otto e mezza": "alle otto e mezza",  # correct
                "a otto e mezza": "alle otto e mezza",
                "in otto e mezza": "alle otto e mezza",
                "nelle otto": "alle otto",
                "verso le otto": "verso le otto",  # correct
                "circa otto": "verso le otto",
                "durante tre ore": "per tre ore",  # duration
                "in tre ore": "in tre ore",  # correct - within
                "da tre ore": "da tre ore",  # correct - for (ongoing)
                "fra due giorni": "fra due giorni",  # correct
                "in due giorni": "in due giorni",  # correct - within
                "dopo due giorni": "fra due giorni",  # after = fra for future
            },
            # Weather and impersonal expressions
            "weather_expressions": {
                "è piovendo": "sta piovendo",
                "è nevicando": "sta nevicando",
                "è facendo freddo": "fa freddo",
                "è facendo caldo": "fa caldo",
                "è avendo sole": "c'è il sole",
                "è essendo nuvoloso": "è nuvoloso",
                "è ventoso": "c'è vento",
                "è piovoso": "piove",
                "è soleggiato": "c'è il sole",
            },
            # Quantity expressions
            "quantity_expressions": {
                "un po di": "un po' di",
                "un poco di": "un po' di",
                "qualche volta": "qualche volta",  # correct
                "alcune volta": "alcune volte",
                "pochi volta": "poche volte",
                "molti volte": "molte volte",
                "troppi volta": "troppe volte",
                "tante volta": "tante volte",
                "molto gente": "molta gente",
                "troppi persone": "troppe persone",
                "alcuni persone": "alcune persone",
            },
            # Comparatives and superlatives
            "comparatives": {
                "più meglio": "meglio",  # avoid double comparative
                "più peggio": "peggio",
                "più maggiore": "maggiore",
                "più minore": "minore",
                "il più meglio": "il migliore",
                "il più peggio": "il peggiore",
                "più bravo di": "più bravo di",  # correct
                "più bravo che": "più bravo che",  # correct in context
                "migliore di": "migliore di",  # correct
                "migliore che": "meglio che",  # context dependent
            },
            # Spelling and pronunciation
            "spelling": {
                "perche": "perché",
                "cosi": "così",
                "piu": "più",
                "gia": "già",
                "cioe": "cioè",
                "caffe": "caffè",
                "te": "tè",  # tea vs you
                "universita": "università",
                "facilmente": "facilmente",  # correct
                "difficilmente": "difficilmente",  # correct
                "sopratutto": "soprattutto",
                "daccordo": "d'accordo",
                "un altro": "un altro",  # correct
                "unaltro": "un altro",
                "qualcosa": "qualcosa",  # correct
                "qual cosa": "qualcosa",
                "apposta": "apposta",  # correct
                "a posta": "apposta",
                "davvero": "davvero",  # correct
                "da vero": "davvero",
                "purtroppo": "purtroppo",  # correct
                "per troppo": "purtroppo",
            },
            # Regional/dialectal influences
            "regional_errors": {
                "noi si fa": "noi facciamo",  # Tuscan influence
                "voi si dice": "voi dite",
                "si dice loro": "loro dicono",
                "che bello che è": "com'è bello",  # redundant che
                "cosa che fai": "cosa fai",
                "dove che vai": "dove vai",
                "quando che vieni": "quando vieni",
                "come che stai": "come stai",
            },
        }

    def _initialize_error_patterns(self) -> Dict[ErrorType, List[Tuple[Pattern, str, str]]]:
        """Initialize regex patterns for error detection with corrections."""
        return {
            ErrorType.VERB_CONJUGATION: [
                # Essere errors
                (
                    re.compile(r"\b(io|tu|lui|lei|noi|voi|loro)\s+essere\b", re.IGNORECASE),
                    "Verb conjugation: 'essere' needs to be conjugated",
                    "conjugate_essere",
                ),
                (
                    re.compile(r"\bho\s+essere\b", re.IGNORECASE),
                    "Use 'sono' instead of 'ho essere'",
                    "ho_essere_fix",
                ),
                (
                    re.compile(r"\bsono\s+avere\b", re.IGNORECASE),
                    "Use 'ho' instead of 'sono avere'",
                    "sono_avere_fix",
                ),
                # Andare errors
                (
                    re.compile(r"\b(io|tu|lui)\s+(ando|andi|anda)\b", re.IGNORECASE),
                    "Irregular verb 'andare': io vado, tu vai, lui va",
                    "andare_conjugation",
                ),
                (
                    re.compile(r"\bio\s+va\b", re.IGNORECASE),
                    "First person: 'io vado' not 'io va'",
                    "io_va_fix",
                ),
                # Fare errors
                (
                    re.compile(r"\b(io|tu|lui)\s+(faco|fo|faci|face)\b", re.IGNORECASE),
                    "Irregular verb 'fare': io faccio, tu fai, lui fa",
                    "fare_conjugation",
                ),
                # Venire errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+venire\b", re.IGNORECASE),
                    "Irregular verb 'venire': io vengo, tu vieni, lui viene",
                    "venire_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(veno|veni|vene)\b", re.IGNORECASE),
                    "Irregular verb 'venire' conjugation",
                    "venire_fix",
                ),
                # Volere errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+volere\b", re.IGNORECASE),
                    "Irregular verb 'volere': io voglio, tu vuoi, lui vuole",
                    "volere_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(volio|voli|vole)\b", re.IGNORECASE),
                    "Irregular verb 'volere' conjugation",
                    "volere_fix",
                ),
                (
                    re.compile(r"\bio\s+volo\b", re.IGNORECASE),
                    "Be careful: 'volo' means flight, 'voglio' means want",
                    "volo_voglio_fix",
                ),
                # Potere errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+potere\b", re.IGNORECASE),
                    "Irregular verb 'potere': io posso, tu puoi, lui può",
                    "potere_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(poto|poti|pote)\b", re.IGNORECASE),
                    "Irregular verb 'potere' conjugation",
                    "potere_fix",
                ),
                # Dovere errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+dovere\b", re.IGNORECASE),
                    "Irregular verb 'dovere': io devo, tu devi, lui deve",
                    "dovere_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(dovo|dovi|dove)\b", re.IGNORECASE),
                    "Irregular verb 'dovere' conjugation",
                    "dovere_fix",
                ),
                # Sapere errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+sapere\b", re.IGNORECASE),
                    "Irregular verb 'sapere': io so, tu sai, lui sa",
                    "sapere_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(sapo|sapi|sape)\b", re.IGNORECASE),
                    "Irregular verb 'sapere' conjugation",
                    "sapere_fix",
                ),
                # Bere errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+bere\b", re.IGNORECASE),
                    "Irregular verb 'bere': io bevo, tu bevi, lui beve",
                    "bere_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(bero|beri)\b", re.IGNORECASE),
                    "Irregular verb 'bere' conjugation",
                    "bere_fix",
                ),
                # Dare errors
                (
                    re.compile(r"\b(io|tu|lui|lei)\s+dare\b", re.IGNORECASE),
                    "Irregular verb 'dare': io do, tu dai, lui dà",
                    "dare_conjugation",
                ),
                (
                    re.compile(r"\b(io|tu|lui)\s+(dao|dari|dae)\b", re.IGNORECASE),
                    "Irregular verb 'dare' conjugation",
                    "dare_fix",
                ),
                (
                    re.compile(r"\blui\s+da\b", re.IGNORECASE),
                    "Third person 'dare': 'lui dà' (with accent)",
                    "lui_da_accent",
                ),
                # Past tense auxiliary errors
                (
                    re.compile(
                        r"\bho\s+(andato|venuto|arrivato|partito|nato|morto)\b", re.IGNORECASE
                    ),
                    "Movement/state verbs use 'essere': sono andato, sono venuto",
                    "movement_essere",
                ),
                (
                    re.compile(
                        r"\bsono\s+(mangiato|bevuto|comprato|letto|scritto|detto)\b", re.IGNORECASE
                    ),
                    "Action verbs use 'avere': ho mangiato, ho bevuto",
                    "action_avere",
                ),
                # Past participle errors
                (
                    re.compile(
                        r"\bho\s+(mangiat|beut|pres|vist|facet|dit|scritt)\b", re.IGNORECASE
                    ),
                    "Complete the past participle ending",
                    "past_participle_ending",
                ),
                # Subjunctive confusion
                (
                    re.compile(
                        r"\b(penso|credo|spero|voglio)\s+che\s+(lui|lei)\s+(è|ha|va|fa)\b",
                        re.IGNORECASE,
                    ),
                    "After 'che' use subjunctive: sia, abbia, vada, faccia",
                    "subjunctive_after_che",
                ),
                (
                    re.compile(r"\bbisogna\s+che\s+(io|tu|lui)\s+(vado|vai|va)\b", re.IGNORECASE),
                    "After 'bisogna che' use subjunctive: vada",
                    "bisogna_subjunctive",
                ),
                # Conditional errors
                (
                    re.compile(r"\b(io|tu|lui)\s+(volerei|potesti)\b", re.IGNORECASE),
                    "Conditional forms: vorrei, potresti",
                    "conditional_forms",
                ),
                (
                    re.compile(r"\bse\s+(io|tu|lui)\s+(sarei|avresti|farei)\b", re.IGNORECASE),
                    "In 'if' clauses use subjunctive: se fossi, se avessi",
                    "if_subjunctive",
                ),
            ],
            ErrorType.GENDER_AGREEMENT: [
                # Article-noun disagreement
                (
                    re.compile(
                        r"\bun\s+(casa|macchina|ragazza|pizza|birra|strada|banca|scuola)\b",
                        re.IGNORECASE,
                    ),
                    "Feminine nouns use 'una'",
                    "un_feminine_fix",
                ),
                (
                    re.compile(r"\buna\s+(uomo|ragazzo|libro|telefono|computer)\b", re.IGNORECASE),
                    "Masculine nouns use 'un'",
                    "una_masculine_fix",
                ),
                (
                    re.compile(r"\bla\s+(problema|sistema|programma|teorema)\b", re.IGNORECASE),
                    "These nouns are masculine: use 'il'",
                    "la_masculine_fix",
                ),
                (
                    re.compile(r"\bil\s+(persona|mano|foto|radio)\b", re.IGNORECASE),
                    "These nouns are feminine: use 'la'",
                    "il_feminine_fix",
                ),
                # Adjective agreement
                (
                    re.compile(
                        r"\b(ragazza|casa|macchina|pizza)\s+(alto|piccolo|rosso|buono)\b",
                        re.IGNORECASE,
                    ),
                    "Adjectives must agree with feminine nouns",
                    "adjective_feminine_agreement",
                ),
                (
                    re.compile(r"\b(uomo|ragazzo)\s+(bella|carina)\b", re.IGNORECASE),
                    "Adjectives must agree with masculine nouns",
                    "adjective_masculine_agreement",
                ),
            ],
            ErrorType.PREPOSITION: [
                # Place prepositions
                (
                    re.compile(r"\bvado\s+in\s+casa\b", re.IGNORECASE),
                    "Use 'vado a casa' for going home",
                    "vado_casa_fix",
                ),
                (
                    re.compile(r"\bsono\s+in\s+casa\b", re.IGNORECASE),
                    "Use 'sono a casa' for being at home",
                    "sono_casa_fix",
                ),
                (
                    re.compile(r"\bvengo\s+da\s+Italia\b", re.IGNORECASE),
                    "Use 'vengo dall'Italia' (from Italy)",
                    "vengo_italia_fix",
                ),
                (
                    re.compile(r"\bvado\s+(nella|nel)\s+(scuola|lavoro)\b", re.IGNORECASE),
                    "Use 'a scuola/al lavoro' for going to school/work",
                    "vado_scuola_lavoro_fix",
                ),
                # Complex prepositions
                (
                    re.compile(r"\bvado\s+da\s+(medico|dottore|professore)\b", re.IGNORECASE),
                    "Use 'dal medico, dal professore' (to/from a person)",
                    "da_person_fix",
                ),
                (
                    re.compile(r"\bvengo\s+da\s+(farmacia|ufficio)\b", re.IGNORECASE),
                    "Use 'dalla farmacia, dall'ufficio' with articles",
                    "da_place_article_fix",
                ),
                (
                    re.compile(r"\bsono\s+da\s+(Milano|Roma|Napoli)\b", re.IGNORECASE),
                    "Origin: 'sono di Milano' not 'sono da Milano'",
                    "origin_preposition_fix",
                ),
                (
                    re.compile(r"\bvado\s+(in|a)\s+(Venezia|Roma|Milano|Bologna)\b", re.IGNORECASE),
                    "Cities use 'a': 'vado a Roma'",
                    "cities_preposition_fix",
                ),
                (
                    re.compile(r"\babito\s+a\s+(Italia|Francia|Germania)\b", re.IGNORECASE),
                    "Countries use 'in': 'abito in Italia'",
                    "countries_preposition_fix",
                ),
                # Transportation prepositions
                (
                    re.compile(r"\bvado\s+con\s+(macchina|treno|aereo|autobus)\b", re.IGNORECASE),
                    "Transportation: 'in macchina, in treno'",
                    "transport_preposition_fix",
                ),
                (
                    re.compile(r"\bvado\s+con\s+piedi\b", re.IGNORECASE),
                    "On foot: 'vado a piedi'",
                    "a_piedi_fix",
                ),
                (
                    re.compile(r"\bvado\s+in\s+piedi\b", re.IGNORECASE),
                    "On foot: 'vado a piedi' not 'in piedi'",
                    "in_piedi_fix",
                ),
                # Articulated prepositions
                (
                    re.compile(r"\bvado\s+a\s+il\s+(cinema|teatro|ristorante)\b", re.IGNORECASE),
                    "Combine preposition + article: 'al cinema'",
                    "articulated_preposition_fix",
                ),
                (
                    re.compile(r"\bvengo\s+da\s+la\s+(stazione|farmacia)\b", re.IGNORECASE),
                    "Combine preposition + article: 'dalla stazione'",
                    "dalla_fix",
                ),
                (
                    re.compile(r"\bstudio\s+in\s+la\s+biblioteca\b", re.IGNORECASE),
                    "Combine: 'nella biblioteca'",
                    "nella_fix",
                ),
                # Time prepositions
                (
                    re.compile(r"\b(nella|nel)\s+(mattina|sera|pomeriggio)\b", re.IGNORECASE),
                    "Use 'di' for times of day: 'di mattina'",
                    "time_preposition_fix",
                ),
                (
                    re.compile(
                        r"\bin\s+(lunedì|martedì|mercoledì|giovedì|venerdì|sabato|domenica)\b",
                        re.IGNORECASE,
                    ),
                    "Use 'di' for days of the week: 'di lunedì'",
                    "day_preposition_fix",
                ),
                (
                    re.compile(r"\ba\s+(otto|nove|dieci)\s+e\s+mezza\b", re.IGNORECASE),
                    "Time: 'alle otto e mezza' with article",
                    "time_article_fix",
                ),
                (
                    re.compile(r"\bdurante\s+\d+\s+(ore|giorni)\b", re.IGNORECASE),
                    "Duration: use 'per' not 'durante'",
                    "duration_preposition_fix",
                ),
                # Verb prepositions
                (
                    re.compile(r"\b(penso|cerco|smetto)\s+per\s+\w+", re.IGNORECASE),
                    "Check preposition with this verb",
                    "verb_preposition_fix",
                ),
                (
                    re.compile(r"\bpenso\s+a\s+fare\b", re.IGNORECASE),
                    "'Pensare di fare' not 'pensare a fare'",
                    "pensare_di_fix",
                ),
                (
                    re.compile(r"\bcerco\s+per\s+trovare\b", re.IGNORECASE),
                    "'Cercare di trovare' not 'cercare per'",
                    "cercare_di_fix",
                ),
                (
                    re.compile(r"\bsmetto\s+per\s+fumare\b", re.IGNORECASE),
                    "'Smettere di fumare' not 'smettere per'",
                    "smettere_di_fix",
                ),
            ],
            ErrorType.WORD_ORDER: [
                (
                    re.compile(r"\b(molto|troppo)\s+(bene|male|difficile)\s+(è)\b", re.IGNORECASE),
                    "Word order: 'è molto bene' not 'molto bene è'",
                    "word_order_fix",
                ),
                (
                    re.compile(r"\bsempre\s+(io|tu|lui|lei|noi|voi|loro)\s+\w+", re.IGNORECASE),
                    "Adverb placement: put 'sempre' after the verb",
                    "sempre_placement",
                ),
                (
                    re.compile(r"\bmai\s+(io|tu|lui|lei|noi|voi|loro)\s+non", re.IGNORECASE),
                    "'Mai' comes after 'non': 'non vado mai'",
                    "mai_placement",
                ),
            ],
            ErrorType.SPELLING: [
                (
                    re.compile(r"\bperche\b", re.IGNORECASE),
                    "Missing accent: 'perché'",
                    "perche_accent",
                ),
                (re.compile(r"\bcosi\b", re.IGNORECASE), "Missing accent: 'così'", "cosi_accent"),
                (re.compile(r"\bpiu\b", re.IGNORECASE), "Missing accent: 'più'", "piu_accent"),
                (re.compile(r"\bgia\b", re.IGNORECASE), "Missing accent: 'già'", "gia_accent"),
                (re.compile(r"\bcioe\b", re.IGNORECASE), "Missing accent: 'cioè'", "cioe_accent"),
                (
                    re.compile(r"\bcaffe\b", re.IGNORECASE),
                    "Missing accent: 'caffè'",
                    "caffe_accent",
                ),
                (
                    re.compile(r"\buniversita\b", re.IGNORECASE),
                    "Missing accent: 'università'",
                    "universita_accent",
                ),
                (
                    re.compile(r"\bsopratutto\b", re.IGNORECASE),
                    "One word: 'soprattutto'",
                    "soprattutto_fix",
                ),
                (
                    re.compile(r"\bdaccordo\b", re.IGNORECASE),
                    "Apostrophe: 'd'accordo'",
                    "daccordo_apostrophe",
                ),
                (
                    re.compile(r"\bunaltro\b", re.IGNORECASE),
                    "Two words: 'un altro'",
                    "unaltro_separation",
                ),
                (
                    re.compile(r"\bqual\s+cosa\b", re.IGNORECASE),
                    "One word: 'qualcosa'",
                    "qualcosa_fix",
                ),
                (re.compile(r"\ba\s+posta\b", re.IGNORECASE), "One word: 'apposta'", "apposta_fix"),
                (re.compile(r"\bda\s+vero\b", re.IGNORECASE), "One word: 'davvero'", "davvero_fix"),
                (
                    re.compile(r"\bper\s+troppo\b", re.IGNORECASE),
                    "One word: 'purtroppo'",
                    "purtroppo_fix",
                ),
            ],
            # Add new error types
            ErrorType.VOCABULARY: [
                # False friends
                (
                    re.compile(r"\bsono\s+embarrassato\b", re.IGNORECASE),
                    "False friend: 'sono imbarazzato'",
                    "embarrassato_fix",
                ),
                (
                    re.compile(r"\bè\s+molto\s+sensitivo\b", re.IGNORECASE),
                    "False friend: 'è molto sensibile'",
                    "sensitivo_fix",
                ),
                (
                    re.compile(r"\blui\s+è\s+grosso\b", re.IGNORECASE),
                    "Better: 'lui è grande' (grosso = thick/coarse)",
                    "grosso_fix",
                ),
                (
                    re.compile(r"\battendo\s+per\s+te\b", re.IGNORECASE),
                    "Better: 'ti aspetto' (attendere is formal)",
                    "attendo_aspetto",
                ),
                (
                    re.compile(r"\brealizza?\s+che\b", re.IGNORECASE),
                    "Better: 'mi rendo conto che' (realize ≠ realizzare)",
                    "realize_fix",
                ),
                (
                    re.compile(r"\battualmente\b", re.IGNORECASE),
                    "Careful: attualmente = currently, not actually",
                    "attualmente_warning",
                ),
                (
                    re.compile(r"\beventualmente\b", re.IGNORECASE),
                    "Careful: eventualmente = eventually, consider 'forse'",
                    "eventualmente_warning",
                ),
                # Pronoun errors
                (
                    re.compile(r"\bio\s+vedo\s+(lui|lei)\b", re.IGNORECASE),
                    "Direct pronouns: 'lo vedo, la vedo'",
                    "direct_pronoun_fix",
                ),
                (
                    re.compile(r"\bio\s+(compro|mangio)\s+(esso|essa)\b", re.IGNORECASE),
                    "Use 'lo/la': 'lo compro, la mangio'",
                    "esso_essa_fix",
                ),
                (
                    re.compile(r"\bvedo\s+te\b", re.IGNORECASE),
                    "Direct pronoun: 'ti vedo'",
                    "te_ti_fix",
                ),
                (
                    re.compile(r"\b(chiamo|conosco)\s+(voi|loro)\b", re.IGNORECASE),
                    "Direct pronouns: 'vi chiamo, li conosco'",
                    "voi_loro_pronoun_fix",
                ),
                (
                    re.compile(r"\bdo\s+a\s+(lui|lei)\b", re.IGNORECASE),
                    "Indirect pronouns: 'gli do, le do'",
                    "indirect_pronoun_fix",
                ),
                (
                    re.compile(r"\b(parlo|scrivo|telefono)\s+a\s+(te|voi|loro)\b", re.IGNORECASE),
                    "Indirect pronouns: 'ti parlo, vi scrivo, gli telefono'",
                    "indirect_complex_fix",
                ),
                # Reflexive verb errors
                (
                    re.compile(r"\bio\s+(chiamo|lavo|vesto)\s+me\b", re.IGNORECASE),
                    "Reflexive: 'mi chiamo, mi lavo, mi vesto'",
                    "reflexive_me_fix",
                ),
                (
                    re.compile(r"\btu\s+vesti\s+te\b", re.IGNORECASE),
                    "Reflexive: 'ti vesti'",
                    "reflexive_te_fix",
                ),
                (
                    re.compile(r"\blui\s+(sveglia|prepara)\s+se\b", re.IGNORECASE),
                    "Reflexive: 'si sveglia, si prepara'",
                    "reflexive_se_fix",
                ),
                (
                    re.compile(r"\bnoi\s+diveriamo\b", re.IGNORECASE),
                    "Reflexive: 'ci divertiamo'",
                    "divertiamo_fix",
                ),
                # Weather expressions
                (
                    re.compile(r"\bè\s+(piovendo|nevicando)\b", re.IGNORECASE),
                    "Weather: 'sta piovendo, sta nevicando'",
                    "weather_progressive_fix",
                ),
                (
                    re.compile(r"\bè\s+facendo\s+(freddo|caldo)\b", re.IGNORECASE),
                    "Weather: 'fa freddo, fa caldo'",
                    "weather_fare_fix",
                ),
                (
                    re.compile(r"\bè\s+(avendo|essendo)\s+sole\b", re.IGNORECASE),
                    "Weather: 'c'è il sole'",
                    "weather_sole_fix",
                ),
                # Quantity expressions
                (
                    re.compile(r"\bun\s+po\s+di\b", re.IGNORECASE),
                    "Apostrophe: 'un po' di'",
                    "un_po_apostrophe",
                ),
                (
                    re.compile(r"\b(alcune|pochi|troppi)\s+volta\b", re.IGNORECASE),
                    "Plural: 'alcune volte, poche volte, troppe volte'",
                    "volta_plural_fix",
                ),
                (
                    re.compile(r"\bmolto\s+gente\b", re.IGNORECASE),
                    "Feminine: 'molta gente'",
                    "molto_gente_fix",
                ),
                (
                    re.compile(r"\b(troppi|alcuni)\s+persone\b", re.IGNORECASE),
                    "Feminine: 'troppe persone, alcune persone'",
                    "persone_agreement_fix",
                ),
                # Comparative errors
                (
                    re.compile(r"\bpiù\s+(meglio|peggio|maggiore|minore)\b", re.IGNORECASE),
                    "Avoid double comparative: just 'meglio, peggio'",
                    "double_comparative_fix",
                ),
                (
                    re.compile(r"\bil\s+più\s+(meglio|peggio)\b", re.IGNORECASE),
                    "Superlative: 'il migliore, il peggiore'",
                    "superlative_fix",
                ),
                # Regional/dialectal influences
                (
                    re.compile(r"\bnoi\s+si\s+(fa|dice)\b", re.IGNORECASE),
                    "Standard Italian: 'noi facciamo, noi diciamo'",
                    "noi_si_fix",
                ),
                (
                    re.compile(r"\bche\s+(bello|cosa|dove|quando)\s+che\b", re.IGNORECASE),
                    "Redundant 'che': 'com'è bello, cosa fai'",
                    "redundant_che_fix",
                ),
            ],
        }

    def get_correction(self, original_text: str, error_type: ErrorType = None) -> Tuple[str, str]:
        """
        Get correction for the given text.

        Returns:
            Tuple of (corrected_text, explanation)
        """
        original_lower = original_text.lower().strip()

        # Search through all correction categories
        for category, corrections in self.corrections.items():
            if original_lower in corrections:
                return corrections[original_lower], f"Correction from {category}"

        # If no direct match, try pattern matching
        if error_type and error_type in self.error_patterns:
            for pattern, explanation, correction_type in self.error_patterns[error_type]:
                if pattern.search(original_text):
                    corrected = self._apply_pattern_correction(
                        original_text, pattern, correction_type
                    )
                    if corrected != original_text:
                        return corrected, explanation

        return original_text, "No correction found"

    def _apply_pattern_correction(self, text: str, pattern: Pattern, correction_type: str) -> str:
        """Apply pattern-based correction."""
        # This would contain the logic to apply specific correction types
        # For now, we'll do simple replacements for key patterns

        corrections_map = {
            # Basic verb fixes
            "ho_essere_fix": lambda t: t.replace("ho essere", "sono"),
            "sono_avere_fix": lambda t: t.replace("sono avere", "ho"),
            "io_va_fix": lambda t: t.replace("io va", "io vado"),
            # Irregular verb conjugations
            "venire_fix": lambda t: re.sub(r"\bio\s+veno\b", "io vengo", t, flags=re.IGNORECASE),
            "volere_fix": lambda t: re.sub(r"\bio\s+volio\b", "io voglio", t, flags=re.IGNORECASE),
            "volo_voglio_fix": lambda t: re.sub(
                r"\bio\s+volo\b", "io voglio", t, flags=re.IGNORECASE
            ),
            "potere_fix": lambda t: re.sub(r"\bio\s+poto\b", "io posso", t, flags=re.IGNORECASE),
            "dovere_fix": lambda t: re.sub(r"\bio\s+dovo\b", "io devo", t, flags=re.IGNORECASE),
            "sapere_fix": lambda t: re.sub(r"\bio\s+sapo\b", "io so", t, flags=re.IGNORECASE),
            "bere_fix": lambda t: re.sub(r"\bio\s+bero\b", "io bevo", t, flags=re.IGNORECASE),
            "dare_fix": lambda t: re.sub(r"\bio\s+dao\b", "io do", t, flags=re.IGNORECASE),
            "lui_da_accent": lambda t: re.sub(r"\blui\s+da\b", "lui dà", t, flags=re.IGNORECASE),
            # Past tense auxiliaries
            "movement_essere": lambda t: re.sub(
                r"\bho\s+(andato|venuto|arrivato|partito|nato)\b",
                lambda m: f"sono {m.group(1)}",
                t,
                flags=re.IGNORECASE,
            ),
            "action_avere": lambda t: re.sub(
                r"\bsono\s+(mangiato|bevuto|comprato|letto|scritto)\b",
                lambda m: f"ho {m.group(1)}",
                t,
                flags=re.IGNORECASE,
            ),
            # Preposition fixes
            "vado_casa_fix": lambda t: t.replace("vado in casa", "vado a casa"),
            "sono_casa_fix": lambda t: t.replace("sono in casa", "sono a casa"),
            "vengo_italia_fix": lambda t: t.replace("vengo da Italia", "vengo dall'Italia"),
            "da_person_fix": lambda t: re.sub(
                r"\bvado\s+da\s+(medico|dottore)\b", r"vado dal \1", t, flags=re.IGNORECASE
            ),
            "da_place_article_fix": lambda t: re.sub(
                r"\bvengo\s+da\s+(farmacia|ufficio)\b",
                lambda m: (
                    f"vengo dalla {m.group(1)}"
                    if m.group(1) == "farmacia"
                    else f"vengo dall'{m.group(1)}"
                ),
                t,
                flags=re.IGNORECASE,
            ),
            "origin_preposition_fix": lambda t: re.sub(
                r"\bsono\s+da\s+(Milano|Roma|Napoli)\b", r"sono di \1", t, flags=re.IGNORECASE
            ),
            "cities_preposition_fix": lambda t: re.sub(
                r"\bvado\s+in\s+(Venezia|Roma|Milano|Bologna)\b",
                r"vado a \1",
                t,
                flags=re.IGNORECASE,
            ),
            "countries_preposition_fix": lambda t: re.sub(
                r"\babito\s+a\s+(Italia|Francia|Germania)\b", r"abito in \1", t, flags=re.IGNORECASE
            ),
            "transport_preposition_fix": lambda t: re.sub(
                r"\bvado\s+con\s+(macchina|treno|aereo)\b", r"vado in \1", t, flags=re.IGNORECASE
            ),
            "a_piedi_fix": lambda t: t.replace("vado con piedi", "vado a piedi"),
            "in_piedi_fix": lambda t: t.replace("vado in piedi", "vado a piedi"),
            "articulated_preposition_fix": lambda t: re.sub(
                r"\bvado\s+a\s+il\s+(cinema|teatro|ristorante)\b",
                r"vado al \1",
                t,
                flags=re.IGNORECASE,
            ),
            "dalla_fix": lambda t: re.sub(
                r"\bvengo\s+da\s+la\s+(stazione|farmacia)\b",
                r"vengo dalla \1",
                t,
                flags=re.IGNORECASE,
            ),
            "nella_fix": lambda t: re.sub(
                r"\bstudio\s+in\s+la\s+biblioteca\b",
                "studio nella biblioteca",
                t,
                flags=re.IGNORECASE,
            ),
            "time_article_fix": lambda t: re.sub(
                r"\ba\s+(otto|nove|dieci)\s+e\s+mezza\b", r"alle \1 e mezza", t, flags=re.IGNORECASE
            ),
            "duration_preposition_fix": lambda t: re.sub(
                r"\bdurante\s+(\d+)\s+(ore|giorni)\b", r"per \1 \2", t, flags=re.IGNORECASE
            ),
            "pensare_di_fix": lambda t: t.replace("penso a fare", "penso di fare"),
            "cercare_di_fix": lambda t: t.replace("cerco per trovare", "cerco di trovare"),
            "smettere_di_fix": lambda t: t.replace("smetto per fumare", "smetto di fumare"),
            # Spelling fixes
            "perche_accent": lambda t: t.replace("perche", "perché"),
            "cosi_accent": lambda t: t.replace("cosi", "così"),
            "piu_accent": lambda t: t.replace("piu", "più"),
            "gia_accent": lambda t: t.replace("gia", "già"),
            "cioe_accent": lambda t: t.replace("cioe", "cioè"),
            "caffe_accent": lambda t: t.replace("caffe", "caffè"),
            "universita_accent": lambda t: t.replace("universita", "università"),
            "soprattutto_fix": lambda t: t.replace("sopratutto", "soprattutto"),
            "daccordo_apostrophe": lambda t: t.replace("daccordo", "d'accordo"),
            "unaltro_separation": lambda t: t.replace("unaltro", "un altro"),
            "qualcosa_fix": lambda t: t.replace("qual cosa", "qualcosa"),
            "apposta_fix": lambda t: t.replace("a posta", "apposta"),
            "davvero_fix": lambda t: t.replace("da vero", "davvero"),
            "purtroppo_fix": lambda t: t.replace("per troppo", "purtroppo"),
            # Vocabulary fixes
            "embarrassato_fix": lambda t: t.replace("embarrassato", "imbarazzato"),
            "sensitivo_fix": lambda t: t.replace("sensitivo", "sensibile"),
            "grosso_fix": lambda t: re.sub(
                r"\blui\s+è\s+grosso\b", "lui è grande", t, flags=re.IGNORECASE
            ),
            "attendo_aspetto": lambda t: t.replace("attendo per te", "ti aspetto"),
            "realize_fix": lambda t: re.sub(
                r"\brealizza?\s+che\b", "mi rendo conto che", t, flags=re.IGNORECASE
            ),
            # Pronoun fixes
            "direct_pronoun_fix": lambda t: re.sub(
                r"\bio\s+vedo\s+(lui|lei)\b",
                lambda m: f"io {'lo' if m.group(1) == 'lui' else 'la'} vedo",
                t,
                flags=re.IGNORECASE,
            ),
            "esso_essa_fix": lambda t: re.sub(
                r"\bio\s+(compro|mangio)\s+(esso|essa)\b",
                lambda m: f"io {'lo' if m.group(2) == 'esso' else 'la'} {m.group(1)}",
                t,
                flags=re.IGNORECASE,
            ),
            "te_ti_fix": lambda t: t.replace("vedo te", "ti vedo"),
            "indirect_pronoun_fix": lambda t: re.sub(
                r"\bdo\s+a\s+(lui|lei)\b",
                lambda m: f"{'gli' if m.group(1) == 'lui' else 'le'} do",
                t,
                flags=re.IGNORECASE,
            ),
            # Reflexive fixes
            "reflexive_me_fix": lambda t: re.sub(
                r"\bio\s+(chiamo|lavo|vesto)\s+me\b", r"io mi \1", t, flags=re.IGNORECASE
            ),
            "reflexive_te_fix": lambda t: re.sub(
                r"\btu\s+vesti\s+te\b", "tu ti vesti", t, flags=re.IGNORECASE
            ),
            "reflexive_se_fix": lambda t: re.sub(
                r"\blui\s+(sveglia|prepara)\s+se\b", r"lui si \1", t, flags=re.IGNORECASE
            ),
            "divertiamo_fix": lambda t: t.replace("noi diveriamo", "noi ci divertiamo"),
            # Weather fixes
            "weather_progressive_fix": lambda t: re.sub(
                r"\bè\s+(piovendo|nevicando)\b", r"sta \1", t, flags=re.IGNORECASE
            ),
            "weather_fare_fix": lambda t: re.sub(
                r"\bè\s+facendo\s+(freddo|caldo)\b", r"fa \1", t, flags=re.IGNORECASE
            ),
            "weather_sole_fix": lambda t: re.sub(
                r"\bè\s+(avendo|essendo)\s+sole\b", "c'è il sole", t, flags=re.IGNORECASE
            ),
            # Quantity fixes
            "un_po_apostrophe": lambda t: t.replace("un po di", "un po' di"),
            "volta_plural_fix": lambda t: re.sub(
                r"\b(alcune|pochi|troppi)\s+volta\b",
                lambda m: f"{'alcune' if m.group(1) == 'alcune' else 'poche' if m.group(1) == 'pochi' else 'troppe'} volte",
                t,
                flags=re.IGNORECASE,
            ),
            "molto_gente_fix": lambda t: t.replace("molto gente", "molta gente"),
            "persone_agreement_fix": lambda t: re.sub(
                r"\b(troppi|alcuni)\s+persone\b",
                lambda m: f"{'troppe' if m.group(1) == 'troppi' else 'alcune'} persone",
                t,
                flags=re.IGNORECASE,
            ),
            # Comparative fixes
            "double_comparative_fix": lambda t: re.sub(
                r"\bpiù\s+(meglio|peggio|maggiore|minore)\b", r"\1", t, flags=re.IGNORECASE
            ),
            "superlative_fix": lambda t: re.sub(
                r"\bil\s+più\s+(meglio|peggio)\b",
                lambda m: f"il {'migliore' if m.group(1) == 'meglio' else 'peggiore'}",
                t,
                flags=re.IGNORECASE,
            ),
            # Regional fixes
            "noi_si_fix": lambda t: re.sub(
                r"\bnoi\s+si\s+(fa|dice)\b",
                lambda m: f"noi {'facciamo' if m.group(1) == 'fa' else 'diciamo'}",
                t,
                flags=re.IGNORECASE,
            ),
            "redundant_che_fix": lambda t: re.sub(
                r"\bche\s+(bello|cosa|dove|quando)\s+che\b",
                lambda m: f"com'è {m.group(1)}" if m.group(1) == "bello" else f"{m.group(1)}",
                t,
                flags=re.IGNORECASE,
            ),
        }

        if correction_type in corrections_map:
            return corrections_map[correction_type](text)

        return text

    def get_training_examples(self) -> List[Tuple[str, str, str]]:
        """
        Generate training examples for LoRA fine-tuning.

        Returns:
            List of (incorrect_text, correct_text, error_type) tuples
        """
        examples = []

        for category, corrections in self.corrections.items():
            for incorrect, correct in corrections.items():
                examples.append((incorrect, correct, category))

        return examples

    def get_contextual_examples(self) -> List[Tuple[str, str, str, str]]:
        """
        Generate contextual examples with full sentences.

        Returns:
            List of (incorrect_sentence, correct_sentence, error_type, explanation) tuples
        """
        return [
            # Basic verb conjugation
            (
                "Io ho essere stanco",
                "Io sono stanco",
                "verb_conjugation",
                "Use 'sono' not 'ho essere'",
            ),
            (
                "Tu essere molto gentile",
                "Tu sei molto gentile",
                "verb_conjugation",
                "Conjugate 'essere'",
            ),
            (
                "Io faco colazione",
                "Io faccio colazione",
                "verb_conjugation",
                "Irregular verb 'fare'",
            ),
            ("Lui da molti soldi", "Lui dà molti soldi", "verb_conjugation", "Add accent: 'dà'"),
            # Irregular verbs
            (
                "Io veno domani",
                "Io vengo domani",
                "verb_conjugation",
                "Irregular verb 'venire': io vengo",
            ),
            (
                "Tu volio andare?",
                "Tu vuoi andare?",
                "verb_conjugation",
                "Irregular verb 'volere': tu vuoi",
            ),
            (
                "Io poto farlo",
                "Io posso farlo",
                "verb_conjugation",
                "Irregular verb 'potere': io posso",
            ),
            (
                "Lei dovo studiare",
                "Lei deve studiare",
                "verb_conjugation",
                "Irregular verb 'dovere': deve",
            ),
            (
                "Io sapo la risposta",
                "Io so la risposta",
                "verb_conjugation",
                "Irregular verb 'sapere': so",
            ),
            (
                "Noi bero vino",
                "Noi beviamo vino",
                "verb_conjugation",
                "Irregular verb 'bere': beviamo",
            ),
            # Past tense auxiliaries
            (
                "Ho andato al cinema",
                "Sono andato al cinema",
                "verb_conjugation",
                "Movement verbs use 'essere'",
            ),
            (
                "Sono mangiato pizza",
                "Ho mangiato pizza",
                "verb_conjugation",
                "Action verbs use 'avere'",
            ),
            (
                "Ho arrivato tardi",
                "Sono arrivato tardi",
                "verb_conjugation",
                "Movement: 'sono arrivato'",
            ),
            (
                "Sono comprato un libro",
                "Ho comprato un libro",
                "verb_conjugation",
                "Action: 'ho comprato'",
            ),
            # Subjunctive
            (
                "Penso che lui è bravo",
                "Penso che lui sia bravo",
                "verb_conjugation",
                "After 'che' use subjunctive: sia",
            ),
            (
                "Spero che tu vai bene",
                "Spero che tu vada bene",
                "verb_conjugation",
                "After 'che' use subjunctive: vada",
            ),
            (
                "Bisogna che io vado",
                "Bisogna che io vada",
                "verb_conjugation",
                "After 'bisogna che' use subjunctive",
            ),
            # Prepositions - basic
            (
                "Vado in casa adesso",
                "Vado a casa adesso",
                "preposition",
                "Use 'a casa' for going home",
            ),
            ("Sono in casa", "Sono a casa", "preposition", "Use 'a casa' for being home"),
            (
                "Vengo da Italia domani",
                "Vengo dall'Italia domani",
                "preposition",
                "Use 'dall'Italia'",
            ),
            # Prepositions - complex
            ("Vado da dottore", "Vado dal dottore", "preposition", "Use article: 'dal dottore'"),
            (
                "Vengo da farmacia",
                "Vengo dalla farmacia",
                "preposition",
                "Use article: 'dalla farmacia'",
            ),
            (
                "Sono da Milano",
                "Sono di Milano",
                "preposition",
                "Origin: 'di Milano' not 'da Milano'",
            ),
            ("Vado in Roma", "Vado a Roma", "preposition", "Cities use 'a': 'a Roma'"),
            ("Abito a Italia", "Abito in Italia", "preposition", "Countries use 'in': 'in Italia'"),
            # Transportation
            (
                "Vado con macchina",
                "Vado in macchina",
                "preposition",
                "Transportation: 'in macchina'",
            ),
            ("Viaggio con treno", "Viaggio in treno", "preposition", "Transportation: 'in treno'"),
            ("Vado con piedi", "Vado a piedi", "preposition", "On foot: 'a piedi'"),
            # Articulated prepositions
            ("Vado a il cinema", "Vado al cinema", "preposition", "Combine: 'a + il = al'"),
            (
                "Vengo da la stazione",
                "Vengo dalla stazione",
                "preposition",
                "Combine: 'da + la = dalla'",
            ),
            (
                "Studio in la biblioteca",
                "Studio nella biblioteca",
                "preposition",
                "Combine: 'in + la = nella'",
            ),
            # Time expressions
            ("A otto e mezza", "Alle otto e mezza", "preposition", "Time: 'alle otto e mezza'"),
            ("Nella mattina", "Di mattina", "preposition", "Time of day: 'di mattina'"),
            ("In lunedì", "Di lunedì", "preposition", "Days: 'di lunedì'"),
            ("Durante tre ore", "Per tre ore", "preposition", "Duration: 'per tre ore'"),
            # Gender agreement - articles
            (
                "La problema è difficile",
                "Il problema è difficile",
                "gender_agreement",
                "'Problema' is masculine",
            ),
            (
                "Ho comprato un casa nuova",
                "Ho comprato una casa nuova",
                "gender_agreement",
                "'Casa' is feminine",
            ),
            ("Una uomo italiano", "Un uomo italiano", "gender_agreement", "'Uomo' is masculine"),
            ("Il mano è grande", "La mano è grande", "gender_agreement", "'Mano' is feminine"),
            # Gender agreement - adjectives
            (
                "La ragazza alto",
                "La ragazza alta",
                "gender_agreement",
                "Feminine adjective: 'alta'",
            ),
            ("Il uomo bella", "L'uomo bello", "gender_agreement", "Masculine adjective: 'bello'"),
            ("Una casa rosso", "Una casa rossa", "gender_agreement", "Feminine adjective: 'rossa'"),
            # Pronouns - direct
            ("Io vedo lui", "Io lo vedo", "vocabulary", "Direct pronoun: 'lo vedo'"),
            ("Vedo te", "Ti vedo", "vocabulary", "Direct pronoun: 'ti vedo'"),
            ("Compro esso", "Lo compro", "vocabulary", "Direct pronoun: 'lo compro'"),
            ("Chiamo voi", "Vi chiamo", "vocabulary", "Direct pronoun: 'vi chiamo'"),
            # Pronouns - indirect
            ("Do a lui", "Gli do", "vocabulary", "Indirect pronoun: 'gli do'"),
            ("Parlo a te", "Ti parlo", "vocabulary", "Indirect pronoun: 'ti parlo'"),
            ("Scrivo a loro", "Gli scrivo", "vocabulary", "Indirect pronoun: 'gli scrivo'"),
            # Reflexive verbs
            ("Io chiamo me Marco", "Io mi chiamo Marco", "vocabulary", "Reflexive: 'mi chiamo'"),
            ("Lui lava se", "Lui si lava", "vocabulary", "Reflexive: 'si lava'"),
            ("Tu vesti te", "Tu ti vesti", "vocabulary", "Reflexive: 'ti vesti'"),
            ("Noi diveriamo", "Noi ci divertiamo", "vocabulary", "Reflexive: 'ci divertiamo'"),
            # Weather expressions
            ("È piovendo", "Sta piovendo", "vocabulary", "Weather: 'sta piovendo'"),
            ("È facendo freddo", "Fa freddo", "vocabulary", "Weather: 'fa freddo'"),
            ("È avendo sole", "C'è il sole", "vocabulary", "Weather: 'c'è il sole'"),
            # Quantity expressions
            ("Un po di acqua", "Un po' di acqua", "spelling", "Apostrophe: 'un po''"),
            ("Alcune volta", "Alcune volte", "vocabulary", "Plural: 'alcune volte'"),
            ("Molto gente", "Molta gente", "vocabulary", "Feminine: 'molta gente'"),
            ("Troppi persone", "Troppe persone", "vocabulary", "Feminine: 'troppe persone'"),
            # Comparatives
            ("Più meglio", "Meglio", "vocabulary", "Avoid double comparative"),
            ("Il più meglio", "Il migliore", "vocabulary", "Superlative: 'il migliore'"),
            # False friends
            ("Sono embarrassato", "Sono imbarazzato", "vocabulary", "False friend: 'imbarazzato'"),
            ("È molto sensitivo", "È molto sensibile", "vocabulary", "False friend: 'sensibile'"),
            ("Realizzo che", "Mi rendo conto che", "vocabulary", "Better: 'mi rendo conto che'"),
            ("Attendo per te", "Ti aspetto", "vocabulary", "Better: 'ti aspetto'"),
            # Spelling
            ("Perche non vieni?", "Perché non vieni?", "spelling", "Add accent: 'perché'"),
            ("Piu forte", "Più forte", "spelling", "Add accent: 'più'"),
            ("Gia fatto", "Già fatto", "spelling", "Add accent: 'già'"),
            ("Universita", "Università", "spelling", "Add accent: 'università'"),
            ("Sopratutto", "Soprattutto", "spelling", "One word: 'soprattutto'"),
            ("Daccordo", "D'accordo", "spelling", "Apostrophe: 'd'accordo'"),
            ("Unaltro", "Un altro", "spelling", "Two words: 'un altro'"),
            ("Da vero", "Davvero", "spelling", "One word: 'davvero'"),
            # Word order
            (
                "Molto bene è questo libro",
                "Questo libro è molto bene",
                "word_order",
                "Adjective after verb",
            ),
            ("Sempre io vado", "Io vado sempre", "word_order", "Adverb after verb: 'vado sempre'"),
            (
                "Mai io non vado",
                "Io non vado mai",
                "word_order",
                "'Mai' after 'non': 'non vado mai'",
            ),
            # Regional/dialectal
            (
                "Noi si fa così",
                "Noi facciamo così",
                "vocabulary",
                "Standard Italian: 'noi facciamo'",
            ),
            ("Che bello che è", "Com'è bello", "vocabulary", "Remove redundant 'che'"),
            ("Cosa che fai?", "Cosa fai?", "vocabulary", "Remove redundant 'che'"),
            ("Dove che vai?", "Dove vai?", "vocabulary", "Remove redundant 'che'"),
        ]


# Create global instance
italian_corrections_db = ItalianCorrectionsDB()
