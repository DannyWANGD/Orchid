"""Conservative four-template diagnostics, independent of the poetry model."""

from .corpus import is_wujue


# Wang Li's canonical five-character patterns. No license to classify all historical variants.
TEMPLATES = {
    "ze_start_no_first_rhyme": ("ZZPPZ", "PPZZP", "PPPZZ", "ZZZPP"),
    "ze_start_first_rhyme": ("ZZZPP", "PPZZP", "PPPZZ", "ZZZPP"),
    "ping_start_no_first_rhyme": ("PPPZZ", "ZZZPP", "ZZPPZ", "PPZZP"),
    "ping_start_first_rhyme": ("PPZZP", "ZZZPP", "ZZPPZ", "PPZZP"),
}


def template_score(lines, book):
    if not is_wujue(lines):
        return {"tone": None, "tone_rhyme": None}
    options = [[book.get(char, set()) for char in line] for line in lines]
    possible = [[{"P" if tone=="ping" else "Z" for tone,_ in sets} for sets in line] for line in options]
    absent = any(not element for line in possible for element in line)
    certain = {name for name,t in TEMPLATES.items()
               if all(choices == {code} for line, patt in zip(possible,t)
                      for choices,code in zip(line,patt))}
    fits = {name for name,t in TEMPLATES.items()
            if all(not choices or code in choices for line,patt in zip(possible,t)
                   for choices,code in zip(line,patt))}
    if not fits:
        tone = "fail"
    elif certain:
        tone = "pass"
    else:
        tone = "unknown"

    def rhyme_fits(name):
        lines_to_check = (0,1,3) if name.endswith("_start_first_rhyme") else (1,3)
        sets = [options[i][-1] for i in lines_to_check]
        known = [s for s in sets if s]
        if known and not any(t=="ping" for t,_ in set.intersection(*known)):
            return "fail"
        chars = [lines[i][-1] for i in lines_to_check]
        if len(set(chars)) != len(chars):
            return "fail"
        if name in certain and len(known)==len(sets) and all(len(s)==1 for s in known):
            return "pass"
        return "unknown"

    checks = [rhyme_fits(name) for name in fits]
    combined = "pass" if "pass" in checks else "unknown" if "unknown" in checks else "fail"
    return {"tone":tone, "tone_rhyme":combined, "tone_possible": sorted(fits),
            "tone_certain": sorted(certain), "missing_tone_mapping": absent}
