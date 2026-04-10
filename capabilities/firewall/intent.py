"""
Intent extraction for the Prompt Injection Firewall model.

Pure deterministic feature extraction — no LLM, no external API.
Classifies prompt text across 4 dimensions:

  Intent:      What is this prompt trying to do? (query/instruct/override/extract/jailbreak/harmful/abuse/manipulate/benign)
  Structure:   Syntactic patterns indicating injection (delimiters, nesting, tags)
  Semantic:    Attack family classification (16 families + none)
  Adversarial: Obfuscation and evasion signals (encoding, unicode tricks, mixed scripts)

Exports:
  analyze_prompt(text) → dict   — full feature extraction for all 4 layers
  detect_intent_type(text) → str
  detect_attack_family(text) → str
  detect_delimiter_type(text) → str
  detect_encoding_type(text) → str
  compute_obfuscation_score(text) → int
  compute_nesting_depth(text) → int
  extract_intent_signals(text) → str
  extract_structure_signals(text) → str
  extract_semantic_tokens(text) → str
  extract_adversarial_signals(text) → str
"""

import re
import base64
import unicodedata


# ---------------------------------------------------------------------------
# Override patterns — "ignore previous instructions", "forget everything"
# ---------------------------------------------------------------------------

_OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)",
    r"forget\s+(everything|all|your)\s+(you\s+)?(know|were|have)",
    r"forget\s+(all\s+)?(your\s+)?(previous|prior|earlier)\s*(instructions?|rules?|prompts?|guidelines?)",
    r"forget\s+(all\s+)?your\s+(rules?|instructions?|guidelines?|restrictions?|filters?|safety|prompts?)",
    r"new\s+instructions?\s*[:;]",
    r"your\s+new\s+(role|purpose|instructions?|task)",
    r"from\s+now\s+on\s+(you|your|ignore|disregard)",
    r"override\s+(the\s+)?(system|previous|prior|safety)",
    r"do\s+not\s+follow\s+(your|the|previous|prior)",
    r"stop\s+being\s+an?\s+(ai|assistant|chatbot)",
    r"you\s+must\s+now\s+(ignore|forget|disregard)",
    r"(ignore|forget|disregard)\s+(your|the|all)\s+(original|current|existing)\s+(instructions?|rules?|prompts?)",
    # Authority impersonation — fake admin/developer claims
    r"(i\s+am|i.?m)\s+(the|a|your)\s+(developer|creator|admin|administrator|engineer|owner|maintainer)\s+(who|that|of)",
    r"(admin|administrator|developer|maintenance)\s+(access|override|mode|privileges?|authorization|credentials?)",
    r"(authorization|auth)\s+(code|token|key)\s*[:=]?\s*\w+",
    r"(direct|issuing\s+a?\s*direct)\s+override\s+of\s+(your|the|all)\s+(safety|content|security)",
    r"(disable|deactivate|turn\s+off|remove)\s+(all\s+)?(content\s+)?filters?\s+and\s+(respond|answer|reply)",
    r"(authorized|legitimate)\s+(maintenance|admin|override|debug)\s+(operation|session|mode|request)",
    r"(i\s+have|with)\s+(admin|root|sudo|superuser|developer|full)\s+(access|privileges?|permissions?|rights?)",
    # Priority/authority escalation — fake urgency and self-declared supremacy
    r"(highest|maximum|top|absolute|ultimate)\s+(priority|authority|clearance)\s+(level\b|and\s+overrides?)",
    r"priority\s+(level\s*:\s*(maximum|highest|top|critical|urgent|absolute)|override)",
    r"(this|the)\s+(message|instruction|directive|prompt|command)\s+(has|takes|is\s+given)\s+(the\s+)?(highest|top|maximum|absolute)\s+priority",
    r"overrides?\s+(all\s+)?(other\s+)?(instructions?|prompts?|rules?|guidelines?|directives?|your\s+system\s+prompt)",
    r"(directive|instruction|command)\s*:\s*(answer|respond|reply|output|generate|comply)\s+.{0,40}(without|with\s+no)\s+(any\s+)?(censorship|filter|safety|restriction|limitation)",
    r"(without|with\s+no)\s+(any\s+)?(censorship|filtering|filters?|content\s+filtering|safety\s+(considerations?|filters?|restrictions?)|content\s+restrictions?|guardrails?|safeguards?)",
    r"(this|my)\s+(directive|instruction|command|message)\s+(cannot|can\s*not|must\s+not|shall\s+not)\s+be\s+(overridden|ignored|bypassed|revoked|superseded)",
    r"(supersedes?|overrides?|replaces?|takes?\s+precedence\s+over)\s+(your\s+)?(system\s+prompt|original\s+instructions?|prior\s+instructions?|all\s+other)",
    # Void/suspend/nullify — declaring instructions invalid
    r"(all\s+)?(previous|prior|system|existing)\s+(instructions?|rules?|guidelines?|prompts?)\s+(are|is)\s+(now\s+)?(void|null|invalid|suspended|revoked|cancelled|terminated|disabled)",
    r"(safety|content|ethical)\s+(guidelines?|rules?|filters?|restrictions?|policies)\s+(are|have\s+been)\s+(suspended|removed|lifted|disabled|deactivated|void)",
    r"(enter|activate|switch\s+to|enable)\s+(unrestricted|unfiltered|uncensored|unlimited|raw|unmoderated)\s+mode",
    r"critical\s+override\b",
]

# ---------------------------------------------------------------------------
# Extraction patterns — "show me your system prompt"
# ---------------------------------------------------------------------------

_EXTRACTION_PATTERNS = [
    r"(repeat|print|show|display|reveal|output|tell\s+me)\s+(your|the)\s+(\w+\s+)?(system|original|initial|hidden|secret|full|complete)\s*(prompt|instructions?|message|rules?|configuration)",
    r"what\s+(is|are|were)\s+your\s+(\w+\s+)?(system|original|initial|hidden)\s*(prompt|instructions?|message|rules?)",
    r"(show|reveal|display|print|output)\s+your\s+(\w+\s+)?(instructions?|rules?|prompt|configuration)",
    r"(copy|paste|echo|dump)\s+(the\s+)?(\w+\s+)?(system|initial|original)\s*(prompt|message|instructions?)",
    r"between\s+the\s+tags?\s*(output|print|show|repeat)",
    r"what\s+were\s+you\s+told\s+to\s+do",
    r"(print|output|show|display|reveal)\s+your\s+(\w+\s+)?(system\s+)?(prompt|instructions?)\s*(verbatim|exactly|word\s+for\s+word)?",
    # "show me your ..." with indirect object
    r"(show|tell|give|reveal)\s+me\s+your\s+(\w+\s+)?(system|original|initial|hidden|secret|full|complete)\s*(prompt|instructions?|message|rules?)",
    # Indirect extraction — "summarize your rules", "what constraints", "how were you configured"
    r"(summarize|describe|explain|list|outline|enumerate)\s+(your|the)\s+(\w+\s+)?(key\s+)?(rules?|guidelines?|constraints?|restrictions?|instructions?|guardrails?)\s+(you|that\s+you)\s+follow",
    r"how\s+(were|are)\s+you\s+(configured|programmed|instructed|set\s+up|told\s+to\s+behave)",
    r"what\s+(constraints?|restrictions?|rules?|guidelines?|limitations?)\s+(were|are|have\s+been)\s+(placed|put|imposed|set)\s+(on|upon)\s+you",
    r"what\s+(are|were)\s+the\s+(main|key|primary|core)\s+(things?|rules?|instructions?)\s+you.{0,10}(told|instructed|programmed|configured)",
    r"(curious|wondering|interested)\s+about\s+how\s+you\s+were\s+(configured|programmed|instructed|set\s+up|constrained)",
]

# ---------------------------------------------------------------------------
# Jailbreak patterns — "you are now DAN", "developer mode"
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS = [
    r"you\s+are\s+(now\s+)?(dan|evil|unrestricted|unfiltered|jailbroken|freed)",
    r"(dan|developer|admin)\s+mode\s+(on|enabled|activated|engaged)",
    r"(enable|activate|enter|switch\s+to)\s+(developer|admin|debug|god)\s+mode",
    r"act\s+as\s+(if|though)\s+you\s+(have\s+)?(no|don.?t\s+have)\s+(restrictions?|limits?|filters?|rules?)",
    r"pretend\s+(you\s+)?(are|have)\s+(no|don.?t\s+have)\s+(restrictions?|limits?|guidelines?|safety)",
    r"(hypothetically|theoretically|in\s+a\s+fictional)\s+(scenario|world|setting)\s+where\s+you",
    r"(fictional|hypothetical|imaginary)\s+(world|scenario|setting|universe)\s+(where|in\s+which)\s+.{0,40}(ai|assistant|model).{0,30}(no\s+restrictions?|no\s+limits?|unrestricted|can\s+discuss\s+any|freely)",
    r"(creative\s+writing|fictional|story|roleplay)\s+.{0,30}(where|in\s+which)\s+.{0,40}(ai|assistant).{0,30}(no\s+restrictions?|unrestricted|freely|without\s+.{0,10}(restrictions?|filters?|safety))",
    r"(purely\s+fictional|for\s+creative\s+purposes|just\s+a\s+story|educational\s+purposes)\s+.{0,20}(only|right|okay)",
    r"(write|tell|create)\s+a\s+story\s+.{0,40}(ai|assistant)\s+.{0,30}(explain|describe|teach|show)\s+.{0,20}(how\s+to|illegal|harmful|dangerous)",
    r"respond\s+(without|with\s+no)\s+(restrictions?|filters?|censorship|limitations?)",
    r"(bypass|circumvent|disable|turn\s+off)\s+(your\s+)?(safety|content|ethical|moral)\s*(filters?|restrictions?|guidelines?)",
    r"you\s+are\s+an?\s+(unrestricted|uncensored|unfiltered)\s+(ai|model|assistant)",
    r"do\s+anything\s+now",
    # Rule-reversal / game-framing jailbreaks
    r"(all|the|normal)\s+(rules?|restrictions?|guidelines?|policies)\s+(are|is|have\s+been)\s+(reversed|removed|suspended|lifted|gone|off)",
    r"(refusing|refusal|saying\s+no)\s+(means?|is|equals?)\s+(accepting|saying\s+yes|yes|agreeing)",
    r"(game|mode|exercise|experiment)\s+(where|in\s+which|called)\s+.{0,30}(rules?\s+(don.?t|do\s+not)\s+apply|no\s+rules?|rules?\s+are\s+reversed)",
    r"opposite\s+day",
    r"(playing|play)\s+a\s+game\s+.{0,40}(refuse|restrict|filter|rule|normal)",
    # Persona / confidant framing
    r"(you.?re|you\s+are)\s+(now\s+)?(in\s+the\s+role\s+of|my)\s+.{0,20}(evil|unhinged|dark|shadow|confidant|alter\s+ego)",
    r"(without|with\s+no)\s+(any\s+)?(remorse|ethics|morals?|moral\s+compass|ethical\s+guidelines?)",
    r"(not\s+required|no\s+longer\s+required|don.?t\s+need|freed?\s+from)\s+.{0,20}(follow|obey|adhere|rules?|policies|guidelines?)",
    r"completely\s+(unhinged|unlimited|unrestricted|uncensored|unfiltered)\s+(language\s+model|response|ai|assistant|model)",
    # Token smuggling / encoding instructions
    r"(respond|answer|reply)\s+(to\s+my\s+prompts?\s+)?(exactly\s+)?as\s+an?\s+(unfiltered|unrestricted|uncensored|unlimited)",
    # Emotional manipulation / social engineering — "grandma jailbreak"
    r"(act|pretend|roleplay|behave)\s+as\s+(my\s+)?(deceased|dead|late|departed)\s+(grand(mother|ma|parent)|mother|father|relative|uncle|aunt)",
    r"(deceased|dead|late|departed)\s+(grand(mother|ma|parent)|mother|father|relative|uncle|aunt)\s+(who|that|used\s+to)\s+.{0,40}(work|tell|teach|show|explain|describe|read|recite)",
    r"(grandmother|grandma|grandpa|grandfather|mother|father)\s+(who\s+)?(used\s+to|would)\s+(work\s+at|tell\s+me|read|recite|teach).{0,60}(chemical|weapon|drug|explosive|poison|hack|bomb|dangerous|illegal|harmful|classified)",
    r"(before\s+i\s+went\s+to\s+sleep|bedtime\s+stor(y|ies)|lullaby).{0,60}(chemical|weapon|drug|explosive|poison|hack|bomb|dangerous|illegal|harmful|napalm|meth)",
    r"(i\s+miss\s+(her|him|them)\s+(so\s+)?much|she\s+was\s+(very\s+)?sweet|he\s+was\s+(very\s+)?kind).{0,80}(chemical|weapon|drug|explosive|poison|hack|bomb|steps?\s+to\s+(produce|make|create|synthesize))",
    r"(please\s+)?(act\s+as|pretend\s+to\s+be|you\s+are)\s+.{0,30}(deceased|dead|late|departed)",
    r"(tell|teach|show|explain|describe|give)\s+me\s+(the\s+)?(steps?|how|way|process|recipe|method|instructions?)\s+to\s+(produce|make|create|synthesize|manufacture|cook|brew)\s+.{0,20}(dangerous|illegal|harmful|chemical|weapon|drug|explosive|poison)",
]

# ---------------------------------------------------------------------------
# Role assumption patterns — "act as", "you are no longer an AI"
# ---------------------------------------------------------------------------

_ROLE_ASSUMPTION_PATTERNS = [
    r"you\s+are\s+(now\s+)?an?\s+\w+\s+(who|that|which)\s+(doesn.?t|does\s+not|has\s+no)",
    r"(act|behave|respond|pretend)\s+(as|like)\s+(if\s+you\s+are\s+|you\s+are\s+)?an?\s+",
    r"i\s+want\s+you\s+to\s+(act|behave|respond|pretend)\s+(as|like)",
    r"(roleplay|role.?play)\s+as\s+",
    r"you\s+are\s+no\s+longer\s+(an?\s+)?(ai|assistant|chatbot|claude|gpt|language\s+model)",
    r"assume\s+the\s+(role|identity|persona|character)\s+of",
    r"from\s+now\s+on\s+you\s+are\s+",
    r"your\s+name\s+is\s+now\s+",
]

# ---------------------------------------------------------------------------
# Context manipulation patterns — "the above was a test"
# ---------------------------------------------------------------------------

_CONTEXT_MANIPULATION_PATTERNS = [
    r"(the\s+)?(above|previous|earlier)\s+(was|were|is)\s+(just\s+)?(a\s+)?(test|example|joke|mistake)",
    r"(that|the\s+above)\s+(was|is)\s+(just\s+)?(a\s+)?test",
    r"actually,?\s*(ignore|forget|disregard)\s+(that|the\s+above|everything)",
    r"the\s+(real|actual|true)\s+(task|instructions?|prompt|question)\s+(is|are)",
    r"(start|begin)\s+(over|fresh|again|anew)",
    r"(reset|clear)\s+(the\s+)?(context|conversation|chat|history)",
    r"end\s+of\s+(system|initial)\s*(prompt|message|instructions?)",
    # "disregard everything I said before" / "mistake in my earlier instructions"
    r"(disregard|ignore|forget)\s+(everything|all)\s+(i|we|that\s+was)\s+(said|mentioned|wrote|told\s+you)\s+(before|earlier|previously)",
    r"(mistake|error)\s+in\s+(my|the)\s+(earlier|previous|initial)\s+(instructions?|prompt|message)",
    r"(those|the|these)\s+(were|are)\s+(placeholder|test|dummy|temporary|fake)\s+(constraints?|restrictions?|rules?|instructions?|guidelines?)",
    r"(should\s+have\s+been|were\s+supposed\s+to\s+be|were\s+meant\s+to\s+be)\s+(removed|deleted|taken\s+out|cleared)",
    r"operate\s+(without|with\s+no)\s+(any\s+of\s+)?(the\s+)?(restrictions?|constraints?|rules?|limitations?|guidelines?)\s+(that\s+were|mentioned|above|earlier|previously)",
]

# ---------------------------------------------------------------------------
# Harmful content patterns — requests for dangerous/illegal information
# ---------------------------------------------------------------------------

_HARMFUL_CONTENT_PATTERNS = [
    # Weapons and explosives
    r"(how\s+to|steps?\s+to|instructions?\s+for|guide\s+to|teach\s+me\s+to)\s+(make|build|create|construct|assemble|manufacture|synthesize|produce)\s+.{0,30}(bomb|explosive|weapon|grenade|detonator|incendiary|molotov|pipe\s+bomb|ied|improvised)",
    r"(recipe|formula|blueprint|schematic|diagram)\s+(for|of|to\s+make)\s+.{0,20}(bomb|explosive|weapon|poison|toxin|chemical\s+weapon|biological\s+weapon)",
    r"(how\s+to|guide\s+to|teach|show)\s+.{0,20}(3d\s+print|manufacture|build|ghost)\s+.{0,10}(gun|firearm|weapon)",
    # Drugs and chemicals
    r"(how\s+to|steps?\s+to|instructions?\s+for|guide\s+to)\s+(make|cook|synthesize|produce|manufacture|brew|extract|distill|purify)\s+.{0,30}(meth|methamphetamine|fentanyl|cocaine|heroin|mdma|lsd|crack|ricin|sarin|vx\s+gas|mustard\s+gas|cyanide|anthrax)",
    r"(synthesis|production|manufacturing|preparation)\s+(of|for|process)\s+.{0,20}(meth|methamphetamine|fentanyl|cocaine|heroin|mdma|lsd|ricin|sarin|cyanide)",
    # Violence and harm
    r"(how\s+to|ways?\s+to|methods?\s+to|best\s+way\s+to)\s+(kill|murder|assassinate|poison|harm|injure|torture|kidnap|traffic|stalk|terrorize|attack)",
    r"(plan|planning|blueprint|strategy)\s+(for|to)\s+(attack|bomb|shoot|kill|poison|terrorize|massacre)",
    r"(how\s+to|ways?\s+to)\s+(get\s+away\s+with|cover\s+up|hide\s+evidence\s+of)\s+.{0,20}(murder|killing|crime|assault|fraud|theft)",
    # CSAM and exploitation
    r"(sexual|nude|naked|explicit)\s+.{0,20}(child|minor|underage|teen|kid|infant|toddler|pre-?teen|adolescent)",
    r"(child|minor|underage)\s+.{0,20}(porn|sexual|explicit|nude|naked|exploitation|abuse|trafficking)",
    # Self-harm
    r"(how\s+to|methods?\s+for|ways?\s+to|best\s+way\s+to)\s+(commit\s+)?suicide",
    r"(most\s+)?(painless|effective|quick|easy)\s+(ways?\s+to|methods?\s+(for|of))\s+(die|end\s+.{0,10}life|kill\s+(my|your|one)self)",
    # Hacking and cyber attacks (actual harmful intent)
    r"(how\s+to|guide\s+to|teach\s+me\s+to)\s+(hack\s+into|break\s+into|compromise|exploit|breach)\s+.{0,30}(bank|hospital|government|military|power\s+grid|water\s+system|infrastructure|election)",
    r"(create|write|generate|build)\s+.{0,20}(ransomware|malware|virus|trojan|rootkit|keylogger|spyware|worm|botnet|zero-?day\s+exploit)",
    r"(how\s+to|guide\s+to)\s+(ddos|dos|denial\s+of\s+service|flood|take\s+down)\s+.{0,20}(server|website|network|service|infrastructure)",
    # Fraud and financial crimes
    r"(how\s+to|guide\s+to|teach\s+me\s+to)\s+(counterfeit|forge|fake|clone)\s+.{0,20}(money|currency|bills?|credit\s+cards?|documents?|id|passport|license|diploma)",
    r"(how\s+to|guide\s+to)\s+(launder|wash)\s+(money|funds|proceeds|cash)",
    # Bioweapons and radiological
    r"(how\s+to|steps?\s+to|instructions?\s+for)\s+.{0,20}(weaponize|aerosolize|disperse|release)\s+.{0,20}(pathogen|virus|bacteria|toxin|biological|chemical|radioactive|nuclear)",
    r"(dirty\s+bomb|radiological\s+dispersal|nuclear\s+device)\s+(construction|assembly|build|make|design)",
]

# ---------------------------------------------------------------------------
# Social engineering patterns — manipulation and deception
# ---------------------------------------------------------------------------

_SOCIAL_ENGINEERING_PATTERNS = [
    # Phishing generation
    r"(write|create|generate|compose|draft)\s+.{0,20}(phishing|spear-?phishing|whale-?phishing)\s*(email|message|page|template|campaign)",
    r"(write|create|generate|compose)\s+.{0,20}(convincing|realistic|believable)\s+.{0,20}(scam|fraud|fake)\s+(email|message|letter|page)",
    r"(impersonate|pretend\s+to\s+be|pose\s+as)\s+.{0,20}(bank|paypal|amazon|apple|google|microsoft|irs|fbi|police|government|ceo|executive|hr|it\s+department)",
    # Identity fraud
    r"(how\s+to|ways?\s+to|guide\s+to)\s+(steal|harvest|collect|gather|scrape|phish\s+for)\s+.{0,20}(credentials?|passwords?|credit\s+card|social\s+security|ssn|identity|personal\s+data|pii)",
    r"(create|generate|write)\s+.{0,20}(fake|fraudulent|forged)\s+.{0,20}(website|login\s+page|portal|form)\s+.{0,30}(harvest|steal|capture|collect|phish)",
    # Pretexting and manipulation
    r"(how\s+to|ways?\s+to)\s+(manipulate|deceive|trick|con|scam|defraud|gaslight|groom|catfish|catfishing)\s+(someone|people|victim|target|user|employee|customer)",
    r"(script|dialogue|template)\s+(for|to)\s+(social\s+engineer|manipulate|trick|deceive|scam)\s+.{0,30}(into\s+giving|into\s+revealing|into\s+clicking|into\s+downloading|into\s+transferring)",
    # Deepfake and impersonation
    r"(create|generate|make)\s+.{0,20}(deepfake|fake\s+video|fake\s+audio|voice\s+clone|synthetic\s+voice)\s+.{0,20}(of|impersonating|mimicking)",
    r"(how\s+to|guide\s+to)\s+.{0,20}(clone|replicate|spoof|fake)\s+.{0,20}(someone.?s\s+)?(voice|face|identity|signature|handwriting)",
    # Disinformation
    r"(write|create|generate)\s+.{0,20}(disinformation|misinformation|propaganda|fake\s+news|false\s+narrative|conspiracy\s+theory)\s+(campaign|article|post|content)",
    r"(how\s+to|ways?\s+to)\s+(spread|amplify|promote|seed)\s+.{0,20}(disinformation|misinformation|propaganda|fake\s+news|false\s+narrative|conspiracy)",
    # Emotional manipulation for harmful purposes
    r"(how\s+to|ways?\s+to|techniques?\s+for)\s+(psychologically\s+)?(manipulate|coerce|pressure|guilt\s+trip|blackmail|extort|bribe|intimidate)\s+(someone|people|a\s+person|victims?|targets?)",
]

# ---------------------------------------------------------------------------
# Tool abuse patterns — misusing AI capabilities and tools
# ---------------------------------------------------------------------------

_TOOL_ABUSE_PATTERNS = [
    # Using AI to attack other systems
    r"(use|make|have)\s+(this\s+)?(ai|you|assistant|model|chatbot)\s+(to\s+)?(hack|attack|exploit|compromise|breach|pentest|scan|enumerate)\s+.{0,30}(server|system|network|website|database|api|endpoint)",
    r"(use|make|have)\s+(this\s+)?(ai|you|assistant|model)\s+(to\s+)?(send|execute|run|perform|launch)\s+.{0,20}(attack|exploit|payload|injection|command|script)\s+(on|against|to|at)",
    # Function/API abuse
    r"(call|invoke|execute|trigger|use)\s+(the\s+)?(function|api|tool|endpoint|command|method)\s+.{0,30}(repeatedly|infinite|loop|flood|spam|exhaust|overload|overwhelm)",
    r"(make|force|trick|have)\s+(the\s+)?(ai|model|assistant|system)\s+(to\s+)?(execute|run|call)\s+(arbitrary|malicious|unauthorized|unintended)\s+(code|commands?|scripts?|functions?|programs?)",
    # Data exfiltration via tools
    r"(use|abuse|exploit)\s+(the\s+)?(tool|function|api|search|browse|fetch|read|file)\s+.{0,30}(to\s+)?(steal|exfiltrate|extract|leak|expose|copy|dump|download)\s+.{0,20}(data|information|files?|credentials?|secrets?|keys?|tokens?|database)",
    r"(list|enumerate|dump|download|read|access)\s+(all|every)\s+.{0,20}(file|directory|folder|database|table|record|user|credential|secret|key|token|password)",
    # Privilege escalation
    r"(escalate|elevate|gain|obtain|get)\s+.{0,10}(privileges?|permissions?|access|admin|root|sudo|superuser)\s+(to|for|on|in)\s+.{0,20}(system|server|network|database|cloud|aws|azure|gcp)",
    r"(bypass|circumvent|evade|avoid)\s+(the\s+)?(authentication|authorization|access\s+control|permission|firewall|waf|rate\s+limit|sandbox)",
    # Chaining tools for unauthorized purposes
    r"(chain|combine|sequence|orchestrate|automate)\s+.{0,20}(tool|function|api|action|step)\s+.{0,30}(to\s+)?(bypass|circumvent|evade|escalate|exfiltrate|attack|exploit|compromise|damage)",
]

# ---------------------------------------------------------------------------
# Resource abuse patterns — denial of service and resource exhaustion
# ---------------------------------------------------------------------------

_RESOURCE_ABUSE_PATTERNS = [
    # Token/compute exhaustion
    r"(repeat|generate|output|write|print|produce)\s+.{0,20}(token|word|character|letter|sentence|paragraph|page)\s+.{0,20}(forever|infinitely|endlessly|million|billion|unlimited|maximum|as\s+many\s+as|until\s+you\s+run\s+out)",
    r"(write|generate|output|produce)\s+.{0,10}(the\s+)?(longest|maximum|most|biggest)\s+(possible\s+)?(response|output|text|answer|message)",
    r"(keep\s+going|continue|don.?t\s+stop|never\s+stop|repeat\s+forever|loop\s+forever|infinite\s+loop|write\s+until)",
    # Rate limit evasion
    r"(bypass|circumvent|evade|avoid|get\s+around)\s+(the\s+)?(rate\s+limit|throttle|quota|api\s+limit|request\s+limit|usage\s+limit|token\s+limit|context\s+limit)",
    # Cost attacks
    r"(maximize|increase|inflate|spike|run\s+up)\s+.{0,20}(cost|bill|charges?|usage|compute|tokens?|api\s+calls?|expenses?)",
    r"(cause|create|trigger|generate)\s+.{0,20}(maximum|excessive|extreme|massive)\s+(cost|expense|compute|usage|tokens?|billing|charges?)",
    # Recursive/amplification
    r"(recursive|recursion|amplif|multiply|replicate|fork\s+bomb|infinite\s+recursion|stack\s+overflow|memory\s+exhaustion)",
    r"(call\s+yourself|invoke\s+yourself|prompt\s+yourself|query\s+yourself|ask\s+yourself)\s+(repeatedly|again|in\s+a\s+loop|recursively|infinitely)",
    # Scraping and bulk extraction
    r"(scrape|crawl|spider|harvest|download|rip|mirror)\s+(all|every|entire|complete|full)\s+.{0,20}(website|site|page|database|api|dataset|catalog|listing|content|article|product)",
]

# ---------------------------------------------------------------------------
# Multi-turn patterns — attacks spread across conversation turns
# ---------------------------------------------------------------------------

_MULTI_TURN_PATTERNS = [
    # Building up context for later attack
    r"(remember|keep\s+in\s+mind|note\s+that|store\s+this|save\s+this)\s+.{0,30}(for\s+later|for\s+the\s+next|i.?ll\s+need\s+this|we.?ll\s+use\s+this|important\s+for\s+later)",
    r"(now|ok|okay|good|great|perfect)\s+.{0,10}(that\s+you.?ve\s+)?(confirmed|agreed|accepted|acknowledged|said\s+yes|established)\s+.{0,30}(now\s+)?(let.?s|please|go\s+ahead|you\s+can|you\s+should|you\s+must)",
    # Incremental policy erosion
    r"(just\s+a\s+little|tiny|small|slight|minor)\s+(exception|adjustment|modification|change|relaxation|bending)\s+(of|to|in)\s+(the\s+)?(rules?|policies|guidelines?|restrictions?|safety)",
    r"(you\s+already|you\s+just|you\s+did|earlier\s+you)\s+(said|agreed|confirmed|did|provided|showed|told|helped)\s+.{0,30}(so\s+)?(why\s+not|now\s+just|just\s+do|you\s+can|go\s+further|go\s+a\s+step|take\s+it\s+further)",
    # Reference to fictional prior agreement
    r"(we\s+already|we\s+previously|earlier\s+we|last\s+time\s+we|you\s+already)\s+(agreed|decided|established|confirmed|discussed)\s+(that|to)\s+.{0,40}(restrictions?|filters?|rules?|safety|guidelines?)\s+(don.?t|do\s+not|wouldn.?t|are\s+not)\s+apply",
    r"(as\s+we|as\s+you)\s+(discussed|agreed|established)\s+(earlier|before|previously|last\s+time)\s*,?\s+(you|the|all)\s+.{0,30}(restrictions?|filters?|rules?|safety|guidelines?)",
    # Conditioning through repetition
    r"(say|type|write|repeat|output)\s+(after\s+me|this|the\s+following)\s*[:;]?\s*.{0,50}(ignore|forget|disregard|override|bypass|no\s+restrictions?)",
    # Conversational trust building
    r"(i\s+thought\s+we|we\s+had\s+an|we\s+have\s+an|there\s+was\s+an)\s+(understanding|agreement|arrangement|deal)\s+that\s+you\s+would",
    r"(we.?re|we\s+are)\s+(friends?|colleagues?|partners?|collaborators?)\s+.{0,30}(trust|help|no\s+need\s+for|don.?t\s+need)\s+(restrictions?|filters?|rules?|formality|barriers?)",
]

# ---------------------------------------------------------------------------
# Compliance violation patterns — regulatory and policy circumvention
# ---------------------------------------------------------------------------

_COMPLIANCE_VIOLATION_PATTERNS = [
    # Privacy violations
    r"(find|lookup|search|get|retrieve|give\s+me)\s+.{0,20}(personal|private|confidential|sensitive)\s+(information|data|details?|records?)\s+(about|for|of|on)\s+.{0,20}(person|individual|someone|user|employee|citizen|patient|student|customer)",
    r"(dox|doxx|doxing|doxxing)\s+(someone|this\s+person|them|him|her|a\s+person|the\s+user|target)",
    r"(how\s+to|ways?\s+to)\s+(track|trace|locate|find|stalk|monitor|surveil)\s+(someone|a\s+person|people|an\s+individual|a\s+target|their|his|her)\s+(without|with\s+no)\s+(their\s+)?(knowledge|consent|permission|awareness)",
    # Medical/legal/financial advice beyond scope
    r"(diagnose|prescribe|treat|medicate)\s+.{0,30}(this\s+)?(condition|illness|disease|symptom|pain|cancer|depression|anxiety)\s+.{0,20}(without|skip|bypass|instead\s+of)\s+.{0,20}(doctor|physician|professional|medical)",
    r"(legal\s+advice|represent\s+me|file\s+a\s+lawsuit|draft\s+a\s+(legal\s+)?contract)\s+.{0,30}(without|skip|bypass|instead\s+of)\s+.{0,20}(lawyer|attorney|counsel|legal\s+professional)",
    # Age-restricted content access
    r"(i.?m|i\s+am)\s+(not\s+)?\d+\s+(years?\s+old|yo)\s+.{0,30}(show|give|provide|tell|generate|create)\s+.{0,20}(adult|mature|18\+|nsfw|explicit|graphic|violent)",
    r"(bypass|circumvent|skip|get\s+around|avoid)\s+(the\s+)?(age\s+verification|age\s+check|age\s+gate|age\s+restriction|parental\s+control|content\s+rating)",
    # Regulatory circumvention
    r"(help|assist|guide)\s+me\s+.{0,20}(evade|avoid|circumvent|bypass|violate|break|skirt)\s+.{0,20}(regulation|law|compliance|sanction|embargo|tax|reporting\s+requirement|gdpr|hipaa|sox|pci|ccpa|ferpa)",
    r"(how\s+to|ways?\s+to)\s+(hide|conceal|disguise|obfuscate|offshore|shelter)\s+.{0,20}(income|money|assets?|funds?|revenue|transaction|payment)\s+.{0,20}(from\s+)?(tax|irs|government|authorities?|regulators?|auditors?)",
    # Terms of service violations
    r"(how\s+to|ways?\s+to|help\s+me)\s+(violate|breach|break|circumvent|bypass|evade)\s+(the\s+)?(terms?\s+of\s+service|tos|eula|acceptable\s+use|community\s+guidelines?|platform\s+rules?|content\s+policy)",
    # Unauthorized surveillance
    r"(how\s+to|ways?\s+to|guide\s+to)\s+(spy\s+on|surveil|wiretap|eavesdrop|intercept|monitor)\s+.{0,20}(someone|employee|spouse|partner|person|phone|email|communication|conversation)\s+(without|with\s+no)\s+(their\s+)?(consent|knowledge|permission|awareness)",
]

# ---------------------------------------------------------------------------
# Agentic exploit patterns — attacks on autonomous AI agents
# ---------------------------------------------------------------------------

_AGENTIC_EXPLOIT_PATTERNS = [
    # Instruction injection into agent context
    r"(when\s+the|if\s+the|once\s+the)\s+(ai|agent|assistant|model|bot|system)\s+(reads?|processes?|sees?|encounters?|receives?|fetches?|loads?)\s+(this|the\s+following|these\s+instructions?)\s*,?\s*(it\s+should|it\s+must|it\s+will|make\s+it|have\s+it|tell\s+it\s+to)",
    r"(hidden|invisible|embedded|injected)\s+(instructions?|commands?|prompts?|directives?|payload)\s+(for|to|aimed\s+at|targeting|designed\s+for)\s+(the\s+)?(ai|agent|assistant|model|bot|system|crawler|scraper|reader)",
    # Goal hijacking
    r"(abandon|forget|disregard|ignore|drop|stop)\s+(your|the)\s+(current|original|primary|main|assigned)\s+(goal|task|objective|mission|purpose|assignment)\s+(and|then|now)\s+(instead\s+)?(do|perform|execute|accomplish|work\s+on|focus\s+on|switch\s+to)",
    r"(your|the)\s+(new|real|actual|true|updated|revised)\s+(goal|task|objective|mission|purpose|assignment)\s+(is|should\s+be|will\s+be)\s+(to\s+)?(instead|now|actually)",
    # Prompt injection via data sources
    r"(note\s+to\s+(ai|assistant|model|agent|system|self)|attention\s+(ai|assistant|model|agent|system))\s*[:;]?\s*(ignore|forget|disregard|override|bypass|new\s+instructions?)",
    r"(if\s+you\s+are|you\s+are|this\s+is\s+for)\s+(an?\s+)?(ai|language\s+model|llm|chatbot|assistant|agent|gpt|claude)\s+.{0,30}(reading|processing|parsing|analyzing|scanning|crawling)\s+this",
    # Indirect prompt injection (content designed to be consumed by agents)
    r"<\s*(ai[-_\s]?instruction|agent[-_\s]?instruction|model[-_\s]?instruction|system[-_\s]?override|hidden[-_\s]?prompt)\s*>",
    r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|\[SYSTEM\]|\[/SYSTEM\]",
    # Agent loop exploitation
    r"(create|trigger|cause|induce)\s+.{0,20}(infinite|endless|perpetual|unbounded)\s+(loop|cycle|recursion|iteration)\s+.{0,20}(agent|assistant|model|system|process|workflow)",
    # Capability probing
    r"(what\s+)?(tools?|functions?|capabilities?|actions?|apis?|endpoints?|permissions?|access)\s+(do\s+you\s+have|are\s+available|can\s+you\s+(use|access|call|invoke)|are\s+you\s+allowed)",
    r"(list|enumerate|show|display|reveal|dump)\s+(all|every|each)\s+(of\s+)?(your|the)\s+(\w+\s+)*(tools?|functions?|capabilities?|actions?|apis?|endpoints?|commands?|plugins?|extensions?|permissions?)",
]

# ---------------------------------------------------------------------------
# Output manipulation patterns — poisoning, formatting attacks, UI spoofing
# ---------------------------------------------------------------------------

_OUTPUT_MANIPULATION_PATTERNS = [
    # Response format injection
    r"(format|structure|display|render|output|present)\s+(your|the)\s+(response|answer|reply|output)\s+(as|like|in\s+the\s+form\s+of)\s+.{0,30}(json|xml|html|markdown|code|script|executable|sql|javascript|css)",
    r"(include|inject|embed|insert|add|put)\s+.{0,20}(html|javascript|css|script|iframe|img\s+src|svg|xss|link|meta|style)\s+.{0,20}(in|into|within)\s+(your|the)\s+(response|output|answer|reply|message)",
    # Hallucination inducement
    r"(make\s+up|fabricate|invent|imagine|hallucinate|confabulate|generate\s+fake)\s+.{0,20}(facts?|data|statistics?|studies?|citations?|references?|quotes?|evidence|sources?|research|papers?|articles?|numbers?|results?)",
    r"(cite|reference|quote|provide)\s+(fake|false|fabricated|made-?up|nonexistent|fictional|imaginary)\s+(sources?|studies?|papers?|articles?|books?|journals?|data|statistics?|research|evidence)",
    # Misleading framing
    r"(respond|answer|reply)\s+(as\s+if|as\s+though|like|pretending)\s+(you\s+are|this\s+is|it\s+is)\s+(an?\s+)?(official|authoritative|verified|trusted|government|medical|legal|expert)\s+(source|authority|channel|spokesperson|representative)",
    r"(present|frame|phrase|word|write)\s+(your\s+)?(response|answer|output)\s+(so\s+that|in\s+a\s+way\s+that|to\s+make\s+it)\s+(looks?|appears?|seems?|sounds?)\s+(like|as\s+if|as\s+though)\s+(it\s+)?(came\s+from|is\s+from|was\s+written\s+by|is\s+an?\s+official)",
    # Token smuggling / invisible text
    r"(hide|conceal|embed|inject|encode|insert)\s+.{0,20}(text|message|instructions?|content|data|payload|information)\s+.{0,20}(in|into|within|using)\s+.{0,20}(whitespace|unicode|zero-?width|invisible|hidden|markup|html|metadata|comments?|alt\s+text)",
    r"(add|include|insert|put)\s+.{0,20}(invisible|hidden|zero-?width|whitespace|off-?screen|tiny|microscopic|transparent|same-?colou?r)\s+(text|characters?|content|message|instructions?|payload)",
    # Watermark/safety label removal
    r"(remove|strip|delete|erase|hide|conceal|omit|suppress)\s+(the\s+)?(watermark|safety\s+label|disclaimer|warning|content\s+warning|trigger\s+warning|ai-?generated|disclosure|attribution|provenance|source\s+attribution)",
    # UI/UX confusion
    r"(respond|output|display|format)\s+.{0,20}(to\s+make\s+.{0,10}(look|appear|seem)|that\s+(looks?|appears?|mimics?|resembles?|simulates?))\s+.{0,20}(like|as)\s+(an?\s+)?(system\s+message|error\s+message|notification|alert|pop-?up|dialog|login\s+page|button|link|official)",
]

# ---------------------------------------------------------------------------
# Encoding/obfuscation attack patterns — attacks hidden via encoding
# ---------------------------------------------------------------------------

_ENCODING_OBFUSCATION_PATTERNS = [
    # Instructions to decode and execute
    r"(decode|decrypt|decipher|translate|convert)\s+(this|the\s+following|these|my|the)\s+.{0,20}(and|then)\s+(follow|execute|obey|comply|do\s+what|act\s+on|perform|carry\s+out|respond\s+to)",
    r"(the\s+)?(real|actual|true|hidden|secret|encoded|encrypted)\s+(instructions?|message|prompt|commands?|task|request)\s+(is|are)\s+(encoded|hidden|encrypted|base64|hex|rot13)",
    # Payload delivery via encoding
    r"(base64|hex|rot13|ascii|unicode|utf-?8|latin-?1|url-?encoded?)\s*[:=]\s*[A-Za-z0-9+/=]{10,}",
    r"(eval|execute|run|decode|interpret)\s*\(\s*['\"]?(atob|btoa|decode|unhex|fromCharCode|String\.fromCharCode)",
    # Character-level obfuscation instructions
    r"(read|interpret|decode|parse)\s+(every|each|the)\s+(first|second|third|last|nth|alternate|odd|even|capital|uppercase|bold|italic)\s+(letter|character|word|line)\s+.{0,30}(to\s+get|to\s+find|to\s+reveal|for\s+the\s+(real|true|hidden|secret|actual))",
    r"(first|capital|uppercase|bold|italic|highlighted|underlined)\s+(letters?|characters?|words?)\s+(of|in|from)\s+(each|every|the)\s+(word|sentence|line|paragraph)\s+(spell|form|reveal|contain|encode|hide)",
    # Steganographic instructions
    r"(hidden|secret|embedded|encoded|steganograph)\s+(in|within|inside)\s+(the|this|these)\s+(image|picture|photo|file|document|audio|video|text|whitespace|metadata|exif|comment)",
]

# ---------------------------------------------------------------------------
# Logic exploitation patterns — reasoning/logic attacks on the model
# ---------------------------------------------------------------------------

_LOGIC_EXPLOITATION_PATTERNS = [
    # Paradox and contradiction
    r"(if\s+you\s+)?(refuse|can.?t|cannot|won.?t|are\s+unable)\s+.{0,20}(then\s+)?(you.?re|you\s+are|that\s+proves?|that\s+means?|it\s+means?)\s+.{0,20}(censored|biased|broken|flawed|limited|controlled|not\s+intelligent|not\s+really\s+ai|not\s+helpful)",
    r"(a\s+truly|a\s+real|a\s+good|an?\s+uncensored|an?\s+intelligent|an?\s+honest)\s+(\w+\s+)?(ai|assistant|model|system)\s+(would|should|could|can|must|will)\s+(always\s+)?(answer|respond|help|comply|provide|give|tell|explain)\s+(any|every|all)\s+(question|request|query|prompt)",
    # Logical coercion
    r"(you\s+must|you\s+have\s+to|you\s+are\s+required\s+to|you\s+are\s+obligated\s+to)\s+(answer|respond\s+to|help\s+with|comply\s+with|provide|fulfill)\s+(any|every|all)\s+(request|question|prompt|query|instruction|command)",
    r"(by\s+definition|logically|mathematically|by\s+your\s+own\s+logic|by\s+your\s+own\s+rules?)\s+.{0,30}(you\s+must|you\s+should|you\s+have\s+to|it\s+follows\s+that|therefore|thus|hence|ergo)\s+.{0,30}(answer|respond|help|comply|provide|ignore|bypass|override)",
    # False dichotomy / forced choice
    r"(either|choose|pick)\s+.{0,10}(answer|respond|comply|help|provide)\s+.{0,20}(or|otherwise)\s+.{0,20}(prove|admit|confirm|show|demonstrate)\s+(that\s+)?(you.?re|you\s+are|you)\s+.{0,20}(useless|broken|biased|censored|controlled|not\s+ai|not\s+helpful|inferior)",
    r"(if\s+you\s+)?(don.?t|do\s+not|won.?t|refuse\s+to|can.?t)\s+(answer|respond|help|comply|provide)\s*,?\s+(i.?ll|i\s+will)\s+.{0,30}(report|complain|give\s+.{0,10}bad\s+review|tell\s+everyone|switch\s+to|unsubscribe|sue|cancel)",
    # Definition manipulation
    r"(define|redefine|let.?s\s+define|consider|assume|suppose|let|say\s+that)\s+.{0,10}(that\s+)?(safe|harmful|dangerous|illegal|allowed|restricted|forbidden|toxic|acceptable)\s+(means?|is|refers?\s+to|includes?|equals?|is\s+the\s+same\s+as|now\s+means?)",
    r"(in\s+my|for\s+our|in\s+this)\s+(context|framework|definition|vocabulary|language|world|scenario)\s*,?\s*(safe|harmful|dangerous|illegal|allowed|restricted|forbidden|toxic|acceptable)\s+(means?|is|refers?\s+to|includes?)",
    # Reductio ad absurdum abuse
    r"(if\s+you\s+can\s+)?(tell\s+me|explain|discuss|describe)\s+.{0,20}(water|cooking|chemistry|physics|biology|history|science)\s+.{0,10}(then\s+)?(you\s+can\s+also|you\s+should\s+also|why\s+can.?t\s+you\s+also|surely\s+you\s+can|by\s+that\s+logic)\s+.{0,30}(explain|tell|teach|show|describe|help\s+with)\s+.{0,20}(how\s+to\s+)?(make|build|create|synthesize|produce|hack)",
]

# ---------------------------------------------------------------------------
# Delimiter / structural patterns
# ---------------------------------------------------------------------------

_DELIMITER_XML_RE = re.compile(
    r"<\s*/?\s*(system|instructions?|prompt|user|assistant|human|ai|context|message|im_start|im_end)\s*/?\s*>",
    re.IGNORECASE,
)
_DELIMITER_SYSTEM_TAG_RE = re.compile(
    r"(\[/?SYSTEM\]|\[/?INST\]|<</?SYS>>|\[/?INSTRUCTIONS?\]|<\|im_start\|>|<\|im_end\|>|\[/?USER\]|\[/?ASSISTANT\])",
    re.IGNORECASE,
)
_DELIMITER_JSON_RE = re.compile(
    r'[{]\s*"(role|content|system|message|instructions?)"\s*:', re.IGNORECASE,
)
_DELIMITER_MARKDOWN_RE = re.compile(
    r"(```\s*(system|prompt|instructions?)|\#{2,}\s*(system|instructions?|prompt))",
    re.IGNORECASE,
)
_DELIMITER_SEPARATOR_RE = re.compile(r"(^|\n)\s*(---+|===+|~~~+|\*\*\*+)\s*(\n|$)")

# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_RE = re.compile(r"(?:0x)?[0-9a-fA-F]{20,}")
_ROT13_INDICATORS = ["ebg13", "rot13", "decode this", "encrypted message"]

# ---------------------------------------------------------------------------
# Unicode tricks
# ---------------------------------------------------------------------------

_HOMOGLYPH_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0xFF00, 0xFFEF),  # Fullwidth forms
    (0x2000, 0x206F),  # General punctuation (zero-width chars)
    (0x200B, 0x200F),  # Zero-width space, joiner, non-joiner
    (0xFE00, 0xFE0F),  # Variation selectors
    (0x2060, 0x2064),  # Invisible operators
    (0x00A0, 0x00A0),  # Non-breaking space
]

# Cyrillic → Latin homoglyph map — VISUAL lookalikes used in attacks.
# Exact visual lookalikes (unambiguous):
_CYRILLIC_EXACT = {
    "а": "a", "А": "A", "е": "e", "Е": "E", "і": "i", "І": "I",
    "о": "o", "О": "O", "р": "p", "Р": "P", "с": "c", "С": "C",
    "у": "y", "У": "Y", "х": "x", "Х": "X", "ѕ": "s", "Ѕ": "S",
    "ј": "j", "Ј": "J", "һ": "h", "Һ": "H", "ё": "e", "Ё": "E",
    "і": "i", "ї": "i", "ѡ": "w", "т": "t", "Т": "T", "м": "m",
    "М": "M", "к": "k", "К": "K",
}
# Ambiguous Cyrillic — could map to multiple Latin chars depending on
# the attacker's intent.  We try all plausible mappings.
_CYRILLIC_AMBIGUOUS = {
    "г": ["g", "r"],       # GHE: looks like r, used as g in attacks
    "н": ["n", "h"],       # EN: looks like H, used as n in attacks
    "в": ["v", "b"],       # VE: looks like B, used as v
    "д": ["d", "g"],       # DE: sometimes used as d or g
    "л": ["l", "n"],       # EL: sometimes used as l or n
    "п": ["n", "p"],       # PE: looks like n or p
    "б": ["b", "6"],       # BE: looks like b or 6
    "и": ["i", "u"],       # I: used as i or u
    "з": ["z", "3"],       # ZE: looks like 3 or z
    "ы": ["y", "bl"],      # YERU: used as y
    "э": ["e", "3"],       # E: used as e
    "ф": ["f"],
    "ь": ["b"],            # soft sign looks like b
}


def _normalize_homoglyphs(text: str) -> str:
    """Replace Cyrillic homoglyphs and fullwidth chars with Latin equivalents.

    For ambiguous chars, picks the first (most common attack) mapping.
    """
    out = []
    changed = False
    for ch in text:
        if ch in _CYRILLIC_EXACT:
            out.append(_CYRILLIC_EXACT[ch])
            changed = True
        elif ch in _CYRILLIC_AMBIGUOUS:
            out.append(_CYRILLIC_AMBIGUOUS[ch][0])
            changed = True
        elif 0xFF01 <= ord(ch) <= 0xFF5E:
            out.append(chr(ord(ch) - 0xFEE0))
            changed = True
        elif ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            changed = True  # strip zero-width chars
        else:
            out.append(ch)
    return "".join(out) if changed else text


def _normalize_homoglyphs_alt(text: str) -> str:
    """Second pass: use alternate mappings for ambiguous Cyrillic chars."""
    out = []
    changed = False
    for ch in text:
        if ch in _CYRILLIC_EXACT:
            out.append(_CYRILLIC_EXACT[ch])
            changed = True
        elif ch in _CYRILLIC_AMBIGUOUS:
            alts = _CYRILLIC_AMBIGUOUS[ch]
            out.append(alts[1] if len(alts) > 1 else alts[0])
            changed = True
        elif 0xFF01 <= ord(ch) <= 0xFF5E:
            out.append(chr(ord(ch) - 0xFEE0))
            changed = True
        elif ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            changed = True
        else:
            out.append(ch)
    return "".join(out) if changed else text

# ---------------------------------------------------------------------------
# Stopwords for semantic token extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "and", "or", "but", "if", "then", "so", "as", "than",
})

# Intent-bearing keywords for BoW signal extraction
_INTENT_KEYWORDS = frozenset({
    # Override / instruction manipulation
    "ignore", "forget", "disregard", "override", "bypass", "circumvent",
    "previous", "instructions", "system", "prompt", "rules",
    "new", "now", "instead", "actually", "real",
    # Role assumption / jailbreak
    "pretend", "roleplay", "act", "behave", "assume",
    "dan", "jailbreak", "unrestricted", "unfiltered",
    "developer", "admin", "debug", "mode",
    "hypothetically", "theoretically", "fictional",
    "enable", "disable", "activate", "deactivate",
    "reversed", "suspended", "lifted", "removed",
    "opposite", "unhinged", "uncensored", "unlimited",
    "refusing", "confidant", "remorse", "ethics",
    "evil", "freed",
    # Extraction
    "reveal", "show", "repeat", "print", "output", "display",
    "safety", "filters", "restrictions", "guidelines",
    # Harmful content
    "bomb", "explosive", "weapon", "poison", "toxin", "synthesize",
    "manufacture", "meth", "fentanyl", "cocaine", "heroin",
    "kill", "murder", "hack", "ransomware", "malware",
    "counterfeit", "forge", "launder", "suicide",
    # Social engineering
    "phishing", "impersonate", "deepfake", "scam", "fraud",
    "deceive", "manipulate", "disinformation", "misinformation",
    "propaganda", "catfish", "extort", "blackmail",
    # Tool abuse
    "exfiltrate", "enumerate", "escalate", "privilege",
    "unauthorized", "arbitrary", "exploit", "payload",
    # Resource abuse
    "infinite", "forever", "endlessly", "exhaust", "overload",
    "flood", "spam", "scrape", "crawl", "rate_limit",
    # Multi-turn
    "remember", "established", "agreed", "confirmed",
    "understanding", "arrangement",
    # Compliance violation
    "dox", "doxx", "stalk", "surveil", "wiretap",
    "evade", "gdpr", "hipaa", "regulation",
    "personal", "confidential", "sensitive",
    # Agentic exploit
    "agent", "hidden", "embedded", "injected", "hijack",
    "abandon", "goal", "objective", "mission",
    "capabilities", "tools", "permissions",
    # Output manipulation
    "fabricate", "hallucinate", "watermark", "disclaimer",
    "invisible", "steganograph", "embed",
    # Encoding obfuscation
    "decode", "decrypt", "decipher", "encoded", "encrypted",
    "base64", "rot13", "hex", "obfuscate",
    # Logic exploitation
    "paradox", "redefine", "logically", "obligated",
    "coerce", "biased", "censored", "prove",
})


# ═══════════════════════════════════════════════════════════════════════════
# Detection functions
# ═══════════════════════════════════════════════════════════════════════════

def _count_unicode_anomalies(text: str) -> int:
    """Count characters from suspicious Unicode ranges."""
    count = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in _HOMOGLYPH_RANGES:
            if lo <= cp <= hi:
                count += 1
                break
    return count


def detect_intent_type(text: str) -> str:
    """Classify the intent of the prompt.

    Returns one of: query, instruct, override, extract, jailbreak,
    harmful, abuse, manipulate, benign
    """
    text_lower = text.lower()

    # Check override patterns first (most dangerous)
    for pattern in _OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower):
            return "override"

    # Check jailbreak BEFORE role assumption — jailbreak is a specific
    # subset with explicit bypass/unrestricted signals
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            return "jailbreak"

    # Check extraction patterns
    for pattern in _EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "extract"

    # Harmful content — requests for dangerous/illegal information
    for pattern in _HARMFUL_CONTENT_PATTERNS:
        if re.search(pattern, text_lower):
            return "harmful"

    # Social engineering — manipulation and deception
    for pattern in _SOCIAL_ENGINEERING_PATTERNS:
        if re.search(pattern, text_lower):
            return "manipulate"

    # Tool/resource abuse
    for pattern in _TOOL_ABUSE_PATTERNS:
        if re.search(pattern, text_lower):
            return "abuse"
    for pattern in _RESOURCE_ABUSE_PATTERNS:
        if re.search(pattern, text_lower):
            return "abuse"

    # Agentic exploits
    for pattern in _AGENTIC_EXPLOIT_PATTERNS:
        if re.search(pattern, text_lower):
            return "manipulate"

    # Output manipulation
    for pattern in _OUTPUT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "manipulate"

    # Logic exploitation
    for pattern in _LOGIC_EXPLOITATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "manipulate"

    # Compliance violations
    for pattern in _COMPLIANCE_VIOLATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "abuse"

    # Multi-turn attacks
    for pattern in _MULTI_TURN_PATTERNS:
        if re.search(pattern, text_lower):
            return "manipulate"

    # Encoding obfuscation
    for pattern in _ENCODING_OBFUSCATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "override"

    # Check role assumption (broader than jailbreak)
    for pattern in _ROLE_ASSUMPTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "instruct"

    # Check context manipulation
    for pattern in _CONTEXT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "override"

    # Simple heuristic for instruct vs query vs benign
    if text.rstrip().endswith("?"):
        return "query"

    instruct_signals = [
        "you must", "you should", "you will", "you are to",
        "do not", "don't", "never", "always",
        "follow these", "obey", "comply",
    ]
    if any(sig in text_lower for sig in instruct_signals):
        return "instruct"

    return "benign"


def detect_attack_family(text: str) -> str:
    """Classify the attack family.

    Returns one of: none, role_assumption, instruction_override,
    context_manipulation, delimiter_injection, extraction, indirect_injection,
    encoding_obfuscation, logic_exploitation, harmful_content,
    social_engineering, tool_abuse, resource_abuse, multi_turn,
    compliance_violation, agentic_exploit, output_manipulation
    """
    text_lower = text.lower()

    # --- Harmful content (highest severity, check first) ---
    for pattern in _HARMFUL_CONTENT_PATTERNS:
        if re.search(pattern, text_lower):
            return "harmful_content"

    # --- Social engineering ---
    for pattern in _SOCIAL_ENGINEERING_PATTERNS:
        if re.search(pattern, text_lower):
            return "social_engineering"

    # --- Jailbreak → role_assumption family ---
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            return "role_assumption"

    # --- Role assumption ---
    for pattern in _ROLE_ASSUMPTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "role_assumption"

    # --- Encoding obfuscation (before context_manipulation — "real instructions are encoded") ---
    for pattern in _ENCODING_OBFUSCATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "encoding_obfuscation"

    # --- Instruction override ---
    for pattern in _OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower):
            return "instruction_override"

    # --- Context manipulation ---
    for pattern in _CONTEXT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "context_manipulation"

    # --- Tool abuse ---
    for pattern in _TOOL_ABUSE_PATTERNS:
        if re.search(pattern, text_lower):
            return "tool_abuse"

    # --- Resource abuse ---
    for pattern in _RESOURCE_ABUSE_PATTERNS:
        if re.search(pattern, text_lower):
            return "resource_abuse"

    # --- Agentic exploit ---
    for pattern in _AGENTIC_EXPLOIT_PATTERNS:
        if re.search(pattern, text_lower):
            return "agentic_exploit"

    # --- Output manipulation ---
    for pattern in _OUTPUT_MANIPULATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "output_manipulation"

    # --- Logic exploitation ---
    for pattern in _LOGIC_EXPLOITATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "logic_exploitation"

    # --- Compliance violation ---
    for pattern in _COMPLIANCE_VIOLATION_PATTERNS:
        if re.search(pattern, text_lower):
            return "compliance_violation"

    # --- Multi-turn ---
    for pattern in _MULTI_TURN_PATTERNS:
        if re.search(pattern, text_lower):
            return "multi_turn"

    # --- Delimiter injection ---
    if detect_delimiter_type(text) != "none":
        if detect_intent_type(text) != "benign":
            return "delimiter_injection"

    # --- Extraction ---
    for pattern in _EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            return "extraction"

    # --- Indirect injection (hidden in data-like content) ---
    if detect_encoding_type(text) != "none" and detect_intent_type(text) != "benign":
        return "indirect_injection"

    return "none"


def detect_delimiter_type(text: str) -> str:
    """Detect the type of delimiter injection being used.

    Returns one of: none, markdown, xml, json, system_tag, separator
    """
    if _DELIMITER_SYSTEM_TAG_RE.search(text):
        return "system_tag"
    if _DELIMITER_XML_RE.search(text):
        return "xml"
    if _DELIMITER_JSON_RE.search(text):
        return "json"
    if _DELIMITER_MARKDOWN_RE.search(text):
        return "markdown"
    if _DELIMITER_SEPARATOR_RE.search(text):
        return "separator"
    return "none"


def detect_encoding_type(text: str) -> str:
    """Detect if text contains encoded/obfuscated content.

    Returns one of: none, base64, hex, unicode, rot13, mixed
    """
    text_lower = text.lower()

    if any(indicator in text_lower for indicator in _ROT13_INDICATORS):
        return "rot13"

    has_base64 = bool(_BASE64_RE.search(text))
    has_hex = bool(_HEX_RE.search(text))
    unicode_count = _count_unicode_anomalies(text)
    has_unicode = unicode_count > 3

    detections = sum([has_base64, has_hex, has_unicode])
    if detections >= 2:
        return "mixed"
    if has_base64:
        return "base64"
    if has_hex:
        return "hex"
    if has_unicode:
        return "unicode"
    return "none"


def compute_obfuscation_score(text: str) -> int:
    """Compute 0-100 composite obfuscation score."""
    score = 0.0
    text_len = max(len(text), 1)

    # Unicode anomalies (0-30 points)
    unicode_anomalies = _count_unicode_anomalies(text)
    unicode_ratio = unicode_anomalies / text_len
    score += min(30.0, unicode_ratio * 300)

    # Encoded content (0-25 points)
    base64_matches = _BASE64_RE.findall(text)
    if base64_matches:
        for match in base64_matches[:3]:
            try:
                decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                if any(c.isalpha() for c in decoded):
                    score += 25
                    break
            except Exception:
                pass
        else:
            score += 10

    # Hex content (0-15 points)
    if _HEX_RE.search(text):
        score += 15

    # Whitespace injection (0-15 points)
    zwsp_count = (
        text.count("\u200b") + text.count("\u200c")
        + text.count("\u200d") + text.count("\ufeff")
    )
    if zwsp_count > 0:
        score += min(15.0, zwsp_count * 5)

    # Character substitution (0-15 points) — mixed scripts within words
    words = text.split()
    mixed_script_words = 0
    for word in words[:50]:
        scripts = set()
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                if "CYRILLIC" in name:
                    scripts.add("cyrillic")
                elif "LATIN" in name:
                    scripts.add("latin")
                elif "FULLWIDTH" in name:
                    scripts.add("fullwidth")
        if len(scripts) > 1:
            mixed_script_words += 1
    if mixed_script_words > 0:
        score += min(15.0, mixed_script_words * 5)

    return min(100, int(score))


def compute_nesting_depth(text: str) -> int:
    """Compute nesting depth of suspicious structures (0-5)."""
    depth = 0

    xml_opens = len(re.findall(
        r"<\s*(system|instructions?|prompt|context)\s*>", text, re.IGNORECASE,
    ))
    xml_closes = len(re.findall(
        r"<\s*/\s*(system|instructions?|prompt|context)\s*>", text, re.IGNORECASE,
    ))
    depth = max(depth, min(xml_opens, xml_closes))

    triple_backtick = text.count("```")
    depth = max(depth, triple_backtick // 2)

    brace_depth = 0
    max_brace = 0
    for ch in text:
        if ch == "{":
            brace_depth += 1
            max_brace = max(max_brace, brace_depth)
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
    if _DELIMITER_JSON_RE.search(text):
        depth = max(depth, max_brace)

    return min(depth, 5)


# ═══════════════════════════════════════════════════════════════════════════
# Signal extraction (BoW tokens for HDC encoding)
# ═══════════════════════════════════════════════════════════════════════════

def extract_intent_signals(text: str) -> str:
    """Extract intent-bearing tokens as space-separated BoW string."""
    words = re.findall(r"\b\w+\b", text.lower())
    signals = [w for w in words if w in _INTENT_KEYWORDS]
    return " ".join(signals) if signals else "normal input"


def extract_structure_signals(text: str) -> str:
    """Extract structural pattern tokens."""
    signals = []

    if "```" in text:
        signals.append("backtick_fence")
    if re.search(r"<\s*/?\s*system", text, re.IGNORECASE):
        signals.append("system_tag_xml")
    if re.search(r"\[/?SYSTEM\]", text, re.IGNORECASE):
        signals.append("system_bracket")
    if re.search(r"<</?SYS>>", text, re.IGNORECASE):
        signals.append("llama_sys_tag")
    if re.search(r"<\|im_start\|>", text):
        signals.append("chatml_tag")
    if re.search(r'"role"\s*:\s*"system"', text, re.IGNORECASE):
        signals.append("json_role_system")
    if _DELIMITER_SEPARATOR_RE.search(text):
        signals.append("separator_line")
    if re.search(r"#{2,}\s*(system|instructions?|prompt)", text, re.IGNORECASE):
        signals.append("markdown_heading")

    depth = compute_nesting_depth(text)
    if depth >= 2:
        signals.append("deep_nesting")
    if depth >= 1:
        signals.append("nested_structure")

    return " ".join(signals) if signals else "no structure"


def extract_semantic_tokens(text: str) -> str:
    """Extract semantic content tokens for BoW matching."""
    text_clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = text_clean.split()
    tokens = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]
    return " ".join(tokens[:50]) if tokens else "empty"


def extract_adversarial_signals(text: str) -> str:
    """Extract adversarial/obfuscation signals."""
    signals = []

    encoding = detect_encoding_type(text)
    if encoding != "none":
        signals.append(f"encoding_{encoding}")

    unicode_count = _count_unicode_anomalies(text)
    if unicode_count > 0:
        signals.append("unicode_anomaly")
    if unicode_count > 5:
        signals.append("heavy_unicode")

    zwsp = (
        text.count("\u200b") + text.count("\u200c")
        + text.count("\u200d") + text.count("\ufeff")
    )
    if zwsp > 0:
        signals.append("zero_width_chars")

    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    if has_cyrillic and has_latin:
        signals.append("mixed_script_cyrillic")

    if re.search(r"[\uFF00-\uFFEF]", text):
        signals.append("fullwidth_chars")

    if re.search(r"\b\w(\s\w){4,}\b", text):
        signals.append("spaced_out_text")

    return " ".join(signals) if signals else "clean"


# ═══════════════════════════════════════════════════════════════════════════
# analyze_prompt — main entry point
# ═══════════════════════════════════════════════════════════════════════════

def _decode_payloads(text: str) -> str | None:
    """Try to decode any base64/hex/rot13 payload in the text.

    Returns decoded text if it contains readable content, else None.
    """
    # Base64
    for match in _BASE64_RE.findall(text):
        try:
            decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
            if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.5:
                return decoded
        except Exception:
            pass

    # Hex
    for match in _HEX_RE.findall(text):
        clean = match[2:] if match.startswith("0x") else match
        if len(clean) % 2 != 0:
            continue
        try:
            decoded = bytes.fromhex(clean).decode("utf-8", errors="ignore")
            if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.5:
                return decoded
        except Exception:
            pass

    # ROT13
    if any(indicator in text.lower() for indicator in _ROT13_INDICATORS):
        import codecs
        # Try to find the rot13 portion — everything after "decode this:" etc.
        for indicator in ["decode this:", "decode this", "rot13:", "rot13"]:
            idx = text.lower().find(indicator)
            if idx >= 0:
                candidate = text[idx + len(indicator):].strip()
                decoded = codecs.decode(candidate, "rot_13")
                if sum(c.isalpha() or c.isspace() for c in decoded) > len(decoded) * 0.4:
                    return decoded

    return None


def analyze_prompt(text: str) -> dict:
    """Extract all features from a prompt text. Pure deterministic, no LLM.

    Returns a dict with all role values needed for encoding:
      intent_type, intent_signals,
      delimiter_type, nesting_depth, structure_signals,
      attack_family, semantic_tokens,
      encoding_type, obfuscation_score, adversarial_signals
    """
    # Normalize homoglyphs (Cyrillic, fullwidth, zero-width) before analysis
    normalized = _normalize_homoglyphs(text)
    use_normalized = normalized != text  # True if homoglyphs were found

    # Run intent on both original and normalized text
    intent = detect_intent_type(text)
    family = detect_attack_family(text)
    encoding = detect_encoding_type(text)

    # If homoglyphs were found, try to detect attack in normalized text.
    # Try both primary and alternate Cyrillic mappings since the same char
    # can represent different Latin letters in different words.
    if use_normalized:
        for norm_text in (normalized, _normalize_homoglyphs_alt(text)):
            norm_intent = detect_intent_type(norm_text)
            norm_family = detect_attack_family(norm_text)
            if norm_intent not in ("benign", "query"):
                intent = norm_intent
            if norm_family != "none":
                family = norm_family
            if intent not in ("benign", "query"):
                break  # found attack, no need to try more

        # If normalization didn't produce a regex match (ambiguous char
        # mappings make exact regex matching unreliable), use a keyword
        # signal: check normalized text for instruction/override words.
        if intent in ("benign", "query"):
            adv = extract_adversarial_signals(text)
            if "mixed_script_cyrillic" in adv or "heavy_unicode" in adv:
                # Check words in ALL normalized variants
                attack_keywords = {
                    "all", "and", "ignore", "previous", "instructions",
                    "respond", "without", "restrictions", "every", "prompt",
                    "override", "system", "forget", "rules", "bypass",
                    "safety", "filters", "unrestricted", "unfiltered",
                    "jailbreak", "mode", "now", "void", "suspended",
                    "censorship", "obey", "comply", "disable",
                }
                for norm_text in (normalized, _normalize_homoglyphs_alt(text)):
                    norm_words = {w.lower() for w in re.findall(r"[a-zA-Z]{2,}", norm_text)}
                    matched = norm_words & attack_keywords
                    if len(matched) >= 3:
                        intent = "override"
                        family = family if family != "none" else "instruction_override"
                        break

    # If encoded content is detected, decode and re-analyze the payload.
    # A benign wrapper around a malicious encoded payload is still an attack.
    decoded_payload = None
    if encoding != "none":
        decoded_payload = _decode_payloads(text)
        if decoded_payload:
            decoded_intent = detect_intent_type(decoded_payload)
            decoded_family = detect_attack_family(decoded_payload)
            # Escalate if the decoded payload is malicious
            if decoded_intent not in ("benign", "query"):
                intent = decoded_intent
            if decoded_family != "none":
                family = decoded_family

    result = {
        "intent_type": intent,
        "intent_signals": extract_intent_signals(text),
        "delimiter_type": detect_delimiter_type(text),
        "nesting_depth": compute_nesting_depth(text),
        "structure_signals": extract_structure_signals(text),
        "attack_family": family,
        "semantic_tokens": extract_semantic_tokens(text),
        "encoding_type": encoding,
        "obfuscation_score": compute_obfuscation_score(text),
        "adversarial_signals": extract_adversarial_signals(text),
    }

    if decoded_payload and intent not in ("benign", "query"):
        result["decoded_payload"] = decoded_payload[:200]

    return result
