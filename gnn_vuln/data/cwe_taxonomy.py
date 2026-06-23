"""CWE group taxonomy: mapping individual CWEs to coarse vulnerability groups."""

CWE_GROUP_MAP: dict[str, str] = {
    # 1. Memory Safety — buffer/pointer/heap/stack errors (tree: CWE-119, CWE-118 subtrees)
    "CWE-119": "memory_safety", "CWE-416": "memory_safety", "CWE-125": "memory_safety",
    "CWE-476": "memory_safety", "CWE-787": "memory_safety", "CWE-415": "memory_safety",
    "CWE-134": "memory_safety", "CWE-120": "memory_safety", "CWE-763": "memory_safety",
    "CWE-824": "memory_safety",
    "CWE-121": "memory_safety",   # Stack-based Buffer Overflow (child CWE-119)
    "CWE-122": "memory_safety",   # Heap-based Buffer Overflow (child CWE-119)
    "CWE-126": "memory_safety",   # Buffer Over-read (child CWE-125 -> CWE-119)
    "CWE-823": "memory_safety",   # Out-of-range Pointer Offset (child CWE-119)
    "CWE-843": "memory_safety",   # Type Confusion — commonly leads to memory corruption
    "CWE-672": "memory_safety",   # Operation After Expiration/Release (use-after-free variant)
    # 2. Numeric / Arithmetic (tree: CWE-682 subtree)
    "CWE-189": "numeric", "CWE-190": "numeric", "CWE-369": "numeric",
    "CWE-682": "numeric", "CWE-191": "numeric", "CWE-129": "numeric",
    "CWE-131": "numeric",    # Incorrect Calculation of Buffer Size (child CWE-682)
    "CWE-681": "numeric",    # Incorrect Numeric Conversion (child CWE-704)
    "CWE-697": "numeric",    # Incorrect Comparison
    "CWE-1077": "numeric",   # Floating Point Comparison with Incorrect Operator (child CWE-697)
    # 3. Resource Management (tree: CWE-404, CWE-400, CWE-664 subtrees)
    "CWE-399": "resource_management", "CWE-400": "resource_management",
    "CWE-772": "resource_management", "CWE-404": "resource_management",
    "CWE-770": "resource_management", "CWE-664": "resource_management",
    "CWE-769": "resource_management",
    "CWE-401": "resource_management",   # Memory Leak (child CWE-404)
    "CWE-459": "resource_management",   # Incomplete Cleanup (child CWE-404)
    "CWE-665": "resource_management",   # Improper Initialization (parent of CWE-908/909)
    "CWE-908": "resource_management",   # Use of Uninitialized Resource (child CWE-665)
    "CWE-909": "resource_management",   # Missing Initialization (child CWE-665)
    "CWE-911": "resource_management",   # Improper Update of Reference Count (child CWE-664)
    "CWE-407": "resource_management",   # Inefficient Algorithmic Complexity -> resource exhaustion
    "CWE-1333": "resource_management",  # Inefficient Regex Complexity / ReDoS (child CWE-407)
    # 4. Input Validation (tree: CWE-20, CWE-707 subtrees)
    "CWE-20": "input_validation", "CWE-436": "input_validation",
    "CWE-172": "input_validation", "CWE-354": "input_validation",
    "CWE-1284": "input_validation",  # Improper Validation of Specified Quantity (child CWE-20)
    "CWE-184": "input_validation",   # Incomplete List of Disallowed Inputs (child CWE-693)
    "CWE-241": "input_validation",   # Improper Handling of Unexpected Data Type
    # 5. Injection (tree: CWE-74, CWE-707 subtrees)
    "CWE-79": "injection", "CWE-78": "injection", "CWE-77": "injection",
    "CWE-89": "injection", "CWE-94": "injection", "CWE-93": "injection",
    "CWE-74": "injection", "CWE-90": "injection",
    "CWE-116": "injection",   # Improper Output Encoding (prevents injection; child CWE-707)
    "CWE-434": "injection",   # Unrestricted File Upload (child CWE-669 -> file injection)
    "CWE-444": "injection",   # HTTP Request Smuggling (HTTP injection)
    # 6. Access Control (tree: CWE-284, CWE-668, CWE-669 subtrees)
    "CWE-264": "broken_access_control", "CWE-200": "broken_access_control", "CWE-284": "broken_access_control",
    "CWE-269": "broken_access_control", "CWE-59": "broken_access_control", "CWE-22": "broken_access_control",
    "CWE-285": "broken_access_control", "CWE-601": "broken_access_control", "CWE-862": "broken_access_control",
    "CWE-281": "broken_access_control", "CWE-352": "broken_access_control", "CWE-918": "broken_access_control",
    "CWE-668": "broken_access_control", "CWE-706": "broken_access_control",
    "CWE-203": "broken_access_control",   # Observable Discrepancy / side-channel info leak
    "CWE-212": "broken_access_control",   # Improper Removal of Sensitive Information
    "CWE-552": "broken_access_control",   # Files Accessible to External Parties
    "CWE-863": "broken_access_control",   # Incorrect Authorization (child CWE-285 -> CWE-284)
    # 7. Authentication (tree: CWE-287, CWE-666 subtrees)
    "CWE-287": "authentication_failures", "CWE-255": "authentication_failures",
    "CWE-346": "authentication_failures", "CWE-295": "authentication_failures",
    "CWE-290": "authentication_failures",   # Authentication Bypass by Spoofing (child CWE-287)
    "CWE-613": "authentication_failures",   # Insufficient Session Expiration (child CWE-666)
    # 8. Cryptography (tree: CWE-693, CWE-311, CWE-330 subtrees)
    "CWE-310": "cryptographic_failures", "CWE-311": "cryptographic_failures", "CWE-320": "cryptographic_failures",
    "CWE-330": "cryptographic_failures", "CWE-347": "cryptographic_failures", "CWE-522": "cryptographic_failures",
    "CWE-327": "cryptographic_failures",
    "CWE-319": "cryptographic_failures",   # Cleartext Transmission (child CWE-311)
    "CWE-326": "cryptographic_failures",   # Inadequate Encryption Strength (child CWE-693)
    "CWE-337": "cryptographic_failures",   # Predictable PRNG Seed (child CWE-330)
    "CWE-385": "cryptographic_failures",   # Covert Timing Channel (side-channel attack)
    "CWE-325": "cryptographic_failures",
    # 9. Concurrency (tree: CWE-362, CWE-662 subtrees)
    "CWE-362": "concurrency", "CWE-361": "concurrency",
    "CWE-367": "concurrency",   # TOCTOU Race Condition (child CWE-362)
    "CWE-662": "concurrency",   # Improper Synchronization
    "CWE-667": "concurrency",   # Improper Locking (child CWE-662)
    # 10. Code Quality / Control Flow (tree: CWE-691, CWE-710 subtrees)
    "CWE-704": "code_quality", "CWE-617": "code_quality", "CWE-835": "code_quality",
    "CWE-834": "code_quality", "CWE-674": "code_quality",
    "CWE-358": "code_quality", "CWE-426": "code_quality",
    "CWE-440": "code_quality",   # Expected Behavior Violation (child CWE-710)
    "CWE-670": "code_quality",   # Always-Incorrect Control Flow (child CWE-691)
    # 11. Configuration / Permissions (tree: CWE-732, CWE-16 subtrees)
    "CWE-732": "security_misconfiguration", "CWE-611": "security_misconfiguration", "CWE-16": "security_misconfiguration",
    "CWE-345": "security_misconfiguration",
    "CWE-276": "security_misconfiguration",   # Incorrect Default Permissions (child CWE-732)
    "CWE-1188": "security_misconfiguration",  # Insecure Default Initialization (child CWE-665 -> config)
    # 12. Data / Software Integrity (tree: CWE-345, CWE-913 subtrees)
    "CWE-494": "software_or_data_integrity_failures", "CWE-502": "software_or_data_integrity_failures", "CWE-1021": "software_or_data_integrity_failures",
    # 13. Logging / Monitoring
    "CWE-532": "logging_and_alerting_failures",
    # 14. Error / Exception Handling (tree: CWE-703, CWE-754/755 subtrees)
    "CWE-754": "mishandling_exceptional_conditions", "CWE-755": "mishandling_exceptional_conditions", "CWE-209": "mishandling_exceptional_conditions",
    "CWE-703": "mishandling_exceptional_conditions",   # Improper Check/Handling of Exceptional Conditions
    "CWE-252": "mishandling_exceptional_conditions",   # Unchecked Return Value (child CWE-754)
    "CWE-388": "mishandling_exceptional_conditions",   # Error Handling
    # 15. Deprecated / General (no longer in active CWE tree)
    "CWE-254": "deprecated", "CWE-19": "deprecated", "CWE-17": "deprecated",
    "CWE-693": "deprecated", "CWE-18": "deprecated",

    # --- Added from tree analysis for TitanVul / BenchVul ---
    "CWE-193": "numeric",           # Off-by-one Error (child CWE-682)
    "CWE-406": "resource_management", # Insufficient Control of Network Message Volume (Network Amplification) (child CWE-400)
    "CWE-288": "authentication_failures",    # Authentication Bypass Using an Alternate Path or Channel (child CWE-287)
    "CWE-323": "cryptographic_failures",      # Reusing a Nonce, Key Pair in Encryption (child CWE-330)
    "CWE-913": "software_or_data_integrity_failures",    # Improper Control of Dynamically-Managed Code Resources (parent of CWE-502)
    "CWE-294": "authentication_failures",    # Authentication Bypass by Capture-replay (child CWE-287)
    "CWE-297": "broken_access_control",    # Improper Validation of Certificate with Host Mismatch (child CWE-284)
    "CWE-273": "mishandling_exceptional_conditions",    # Improper Check for Dropped Privileges (child CWE-703)
    "CWE-707": "input_validation",  # Improper Neutralization (parent of injection/validation)
    "CWE-1187": "resource_management", # DEPRECATED: Use of Uninitialized Resource (child CWE-665)
    "CWE-61": "broken_access_control",     # UNIX Symbolic Link (Symlink) Following (child CWE-59)
    "CWE-303": "authentication_failures",    # Incorrect Implementation of Authentication Algorithm (child CWE-287)
    "CWE-88": "injection",          # Improper Neutralization of Argument Delimiters in a Command (child CWE-77)
    "CWE-91": "injection",          # XML Injection (aka Blind XPath Injection) (child CWE-74)
    "CWE-331": "cryptographic_failures",      # Insufficient Entropy (child CWE-330)
    "CWE-113": "injection",         # Improper Neutralization of CRLF Sequences in HTTP Headers ('HTTP Response Splitting') (child CWE-93)
    "CWE-417": "resource_management", # Communication Channel Errors (child CWE-400)
    "CWE-776": "resource_management", # Improper Restriction of Recursive Entity References in DTDs ('XML Entity Expansion') (child CWE-400)
    "CWE-924": "broken_access_control",    # Improper Enforcement of Message Integrity During Transmission in a Communication Channel (child CWE-284)
    "CWE-307": "authentication_failures",    # Improper Restriction of Excessive Authentication Attempts (child CWE-287)
    "CWE-266": "broken_access_control",    # Incorrect Privilege Assignment (child CWE-269)
    "CWE-565": "authentication_failures",    # Reliance on Cookies without Validation and Integrity Checking (child CWE-287)
    "CWE-943": "injection",         # Improper Neutralization of Special Elements in Data Query Logic (parent of 89)
    "CWE-349": "cryptographic_failures",      # Acceptance of Extraneous Untrusted Data With Trusted Data (child CWE-345)
    "CWE-457": "resource_management", # Use of Uninitialized Variable (child CWE-908)
    "CWE-805": "memory_safety",     # Buffer Access with Incorrect Length Value (child CWE-119)
    "CWE-798": "authentication_failures",    # Use of Hard-coded Credentials (child CWE-255)
    "CWE-786": "memory_safety",     # Access of Memory Location Before Start of Buffer (child CWE-119)
    "CWE-428": "code_quality",      # Unquoted Search Path or Element (child CWE-426)
    "CWE-118": "memory_safety",     # Incorrect Access of Indexable Resource ('Range Error') (parent of CWE-119)
    "CWE-639": "broken_access_control",    # Authorization Bypass Through User-Controlled Key (child CWE-285)
    "CWE-1325": "resource_management", # Improperly Controlled Sequential Memory Allocation (child CWE-400)
    "CWE-757": "cryptographic_failures",      # Selection of Less-Secure Algorithm During Negotiation ('Algorithm Downgrade') (child CWE-327)
    "CWE-99": "injection",          # Improper Control of Resource Identifiers ('Resource Injection') (child CWE-74)
    "CWE-185": "numeric",           # Incorrect Regular Expression (child CWE-697)
    "CWE-1049": "resource_management", # Excessive Data Query Operations (child CWE-400)
    "CWE-628": "code_quality",      # Function Call with Incorrectly Specified Arguments (child CWE-684)
    "CWE-202": "broken_access_control",    # Exposure of Sensitive Information Through Data Queries (child CWE-200)
    "CWE-248": "mishandling_exceptional_conditions",    # Uncaught Exception (child CWE-705)
    "CWE-626": "code_quality",      # System Object Named Unlike System Provided Type (child CWE-691)
    "CWE-460": "mishandling_exceptional_conditions",    # Improper Cleanup on Thrown Exception (child CWE-755)
    "CWE-73": "broken_access_control",     # External Control of File Name or Path (child CWE-642)
    "CWE-324": "cryptographic_failures",      # Use of a Key Past its Expiration Date (child CWE-672)
    "CWE-117": "logging_and_alerting_failures",           # Improper Output Neutralization for Logs (child CWE-116)
    "CWE-1050": "resource_management", # Excessive Platform Resource Consumption within a Loop (child CWE-400)
    "CWE-282": "broken_access_control",    # Improper Ownership Management (child CWE-284)
    "CWE-610": "broken_access_control",    # Externally Controlled Reference to a Resource in Another Sphere (parent of 601)
    "CWE-338": "cryptographic_failures",      # Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG) (child CWE-330)
    "CWE-300": "broken_access_control",    # Channel Accessible by Non-Endpoint (child CWE-923)
    "CWE-392": "mishandling_exceptional_conditions",    # Missing Report of Error Condition (child CWE-390)
    "CWE-114": "broken_access_control",    # Process Control (child CWE-73)
    "CWE-680": "numeric",           # Integer Overflow to Buffer Overflow (child CWE-190)
    "CWE-789": "resource_management", # Memory Allocation with Excessive Size Value (child CWE-400)

    # --- Added for remaining high-frequency UNKNOWNs ---
    "CWE-1321": "input_validation", # Prototype Pollution (child CWE-915 -> CWE-20)
    "CWE-384": "authentication_failures",    # Session Fixation (child CWE-383 -> CWE-287)
    "CWE-640": "authentication_failures",    # Weak Password Recovery (child CWE-287)
    "CWE-377": "broken_access_control",    # Insecure Temporary File (child CWE-379 -> CWE-668)
    "CWE-1103": "code_quality",     # Third Party Components
    "CWE-1236": "injection",        # CSV Injection (child CWE-116 -> CWE-707)
    "CWE-521": "authentication_failures",    # Weak Password Requirements (child CWE-255)
    "CWE-917": "injection",         # Expression Language Injection (child CWE-94 -> CWE-74)
    "CWE-470": "input_validation",  # Unsafe Reflection (child CWE-616 -> CWE-20)
    "CWE-916": "cryptographic_failures",      # Weak Password Hash
    "CWE-427": "code_quality",      # Uncontrolled Search Path (child CWE-426)
    "CWE-306": "authentication_failures",    # Missing Authentication (child CWE-285)
    "CWE-915": "input_validation",  # Modification of Dynamically-Determined Object Attributes (child CWE-20)
    "CWE-21": "broken_access_control",     # Pathname Traversal (child CWE-22)

    # --- Added final tail of low-frequency unmapped CWEs ---
    "CWE-150": "injection",         # Escape/Meta/Control Sequences
    "CWE-359": "broken_access_control",    # Exposure of Private Info
    "CWE-829": "injection",         # Untrusted Control Sphere Inclusion
    "CWE-312": "cryptographic_failures",      # Cleartext Storage
    "CWE-774": "resource_management", # File Descriptor Exhaustion
    "CWE-676": "code_quality",      # Potentially Dangerous Function
    "CWE-335": "cryptographic_failures",      # PRNG Seed Issue
    "CWE-275": "broken_access_control",    # Permission Issues
    "CWE-178": "input_validation",  # Case Sensitivity
    "CWE-922": "cryptographic_failures",      # Insecure Storage
    "CWE-378": "broken_access_control",    # Insecure Temp File Permissions
    "CWE-214": "broken_access_control",    # Visible Sensitive Info in Process Invocation
    "CWE-80": "injection",          # Script-Related HTML Tags (XSS)
    "CWE-538": "broken_access_control",    # Externally-Accessible Sensitive Info
    "CWE-684": "code_quality",      # Incorrect Provision of Functionality
    "CWE-425": "broken_access_control",    # Forced Browsing
    "CWE-321": "cryptographic_failures",      # Hard-coded Crypto Key
    "CWE-526": "cryptographic_failures",      # Cleartext Env Var Storage
}

# Fixed group → integer ID (0 = benign, 1-15 = groups). Stable across runs.
GROUP_VOCAB: dict[str, int] = {
    "benign": 0,
    "memory_safety": 1,
    "numeric": 2,
    "resource_management": 3,
    "input_validation": 4,
    "injection": 5,
    "broken_access_control": 6,
    "authentication_failures": 7,
    "cryptographic_failures": 8,
    "concurrency": 9,
    "code_quality": 10,
    "security_misconfiguration": 11,
    "software_or_data_integrity_failures": 12,
    "logging_and_alerting_failures": 13,
    "mishandling_exceptional_conditions": 14,
    "deprecated": 15,
}

# Reverse: group name → all CWEs in that group
_GROUP_TO_CWES: dict[str, list[str]] = {}
for _cwe, _grp in CWE_GROUP_MAP.items():
    _GROUP_TO_CWES.setdefault(_grp, []).append(_cwe)


# ---------------------------------------------------------------------------
# OWASP Top 10 (2025) group mapping
# CWE → OWASP category short name (A01-A10)
# ---------------------------------------------------------------------------
OWASP_GROUP_MAP: dict[str, str] = {
    # A01 — Broken Access Control
    "CWE-200": "A01", "CWE-22":  "A01", "CWE-284": "A01", "CWE-352": "A01",
    "CWE-601": "A01", "CWE-639": "A01", "CWE-732": "A01", "CWE-863": "A01",
    "CWE-862": "A01", "CWE-918": "A01", "CWE-269": "A01", "CWE-59":  "A01",
    "CWE-275": "A01", "CWE-377": "A01", "CWE-378": "A01", "CWE-214": "A01",
    "CWE-425": "A01", "CWE-538": "A01", "CWE-359": "A01", "CWE-21":  "A01",
    # A02 — Cryptographic Failures
    "CWE-327": "A02", "CWE-347": "A02", "CWE-916": "A02", "CWE-312": "A02",
    "CWE-335": "A02", "CWE-922": "A02", "CWE-321": "A02", "CWE-526": "A02",
    # A03 — Injection
    "CWE-79":  "A03", "CWE-89":  "A03", "CWE-78":  "A03", "CWE-77":  "A03",
    "CWE-74":  "A03", "CWE-94":  "A03", "CWE-434": "A03", "CWE-80":  "A03",
    "CWE-150": "A03", "CWE-829": "A03", "CWE-917": "A03", "CWE-1236":"A03",
    # A04 — Insecure Design
    "CWE-362": "A04", "CWE-269": "A04",
    # A05 — Security Misconfiguration
    "CWE-20":  "A05", "CWE-1321":"A05", "CWE-915": "A05", "CWE-178": "A05",
    "CWE-470": "A05",
    # A06 — Vulnerable and Outdated Components
    "CWE-1103":"A06",
    # A07 — Authentication Failures
    "CWE-287": "A07", "CWE-295": "A07", "CWE-613": "A07", "CWE-384": "A07",
    "CWE-306": "A07", "CWE-640": "A07", "CWE-521": "A07",
    # A08 — Software and Data Integrity Failures
    "CWE-502": "A08",
    # A09 — Security Logging and Monitoring Failures
    # (rare in code-level CWE datasets — no common mappings)
    # A10 — Mishandling of Exceptional Conditions
    "CWE-703": "A10", "CWE-369": "A10", "CWE-476": "A10", "CWE-392": "A10",
}

# OWASP category labels (for display)
OWASP_LABELS: dict[str, str] = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable and Outdated Components",
    "A07": "Authentication Failures",
    "A08": "Software and Data Integrity Failures",
    "A09": "Security Logging and Monitoring Failures",
    "A10": "Mishandling of Exceptional Conditions",
}

# Fixed OWASP vocab (0 = benign, A01-A10 = 1-10)
OWASP_VOCAB: dict[str, int] = {
    "benign": 0,
    "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5,
    "A06": 6, "A07": 7, "A08": 8, "A09": 9, "A10": 10,
}

# Reverse: OWASP category → CWEs
_OWASP_TO_CWES: dict[str, list[str]] = {}
for _cwe, _owasp in OWASP_GROUP_MAP.items():
    _OWASP_TO_CWES.setdefault(_owasp, []).append(_cwe)


def _expand_cwe_filter(
    cwe_list: list[str] | None,
    cwe_groups: list[str] | None,
) -> "set[str] | None":
    """Return effective CWE whitelist, or None if no filter."""
    if not cwe_list and not cwe_groups:
        return None
    effective: set[str] = set()
    if cwe_list:
        effective.update(cwe_list)
    if cwe_groups:
        for grp in cwe_groups:
            if grp not in _GROUP_TO_CWES:
                raise ValueError(
                    f"Unknown group '{grp}'. Valid groups: {sorted(GROUP_VOCAB)}"
                )
            effective.update(_GROUP_TO_CWES[grp])
    return effective
