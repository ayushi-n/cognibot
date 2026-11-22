"""REASON: a small forward-chaining knowledge base of IF-THEN rules."""

from . import config as C

RULES = {
    "R1": "IF smoke(c) AND heat(c) THEN likely_fire(c)            [danger 0.9]",
    "R2": "IF burned_at(c) in episodic memory THEN fire(c)        [danger 1.0]",
    "R3": "IF smoke(c) AND no heat reading THEN possible_fire(c)  [danger = P(fire|smoke)]",
    "R4": "IF smoke(c) AND heat absent >= 2 times THEN probably_safe(c)",
    "R5": "IF health < 40 THEN prefer RETREAT",
    "R6": "IF battery < 1.3 x distance_to_base + 8 THEN must RETREAT",
    "R7": "IF survivor known THEN RESCUE is a candidate goal",
    "R8": "IF all reported survivors rescued THEN mission complete -> RETREAT",
}


class KnowledgeBase:
    rules = RULES

    def infer_danger(self, cmap, episodic, semantic):
        """Return danger (0..1) for every smoky cell plus the rules that fired."""
        p = semantic.p_fire_given_smoke()
        burned = episodic.cells_of("burned") | cmap.fire
        danger, self.cell_rule, count = {}, {}, {"R1": 0, "R2": 0, "R3": 0, "R4": 0}
        for c in cmap.smoke:
            if c in burned:
                danger[c], k = 1.0, "R2"
            elif cmap.heat_pos[c] > 0 and cmap.heat_pos[c] >= cmap.heat_neg[c]:
                danger[c], k = 0.9, "R1"
            elif cmap.heat_neg[c] >= 2:
                danger[c], k = p * 0.3, "R4"
            else:
                danger[c], k = p, "R3"
            count[k] += 1
            self.cell_rule[c] = k
        fired = [f"{k}: {v} cell(s)" for k, v in count.items() if v]
        return danger, fired

    def infer_goals(self, health, battery, base_cost, cmap, rescued=0, reported=C.N_SURVIVORS):
        flags = {
            "all_rescued": rescued >= reported,
            "low_health": health < 40,
            "low_battery": base_cost is not None and battery < 1.3 * base_cost + 8,
            "survivor_known": bool(cmap.survivors),
        }
        fired = []
        if flags["all_rescued"]:
            fired.append("R8: all survivors saved -> RETREAT")
        if flags["low_health"]:
            fired.append("R5: health low -> prefer RETREAT")
        if flags["low_battery"]:
            fired.append("R6: battery low -> must RETREAT")
        if flags["survivor_known"]:
            fired.append(f"R7: {len(cmap.survivors)} survivor(s) known -> RESCUE")
        return flags, fired
