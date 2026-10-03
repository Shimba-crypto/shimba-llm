#!/usr/bin/env python3
"""
gen_spike_extra.py -- 1000 math + 1000 reasoning turns for spike.

Every turn is COMPUTED, never templated with a guessed answer:
  math      add / sub / mul / exact-div with worked thinking
  reasoning valid syllogisms, modus ponens/tollens, either-or
            elimination, 4-number sorts -- all derived from the entities

Same 3-line shape as flafi (User:/Thinking:/Response:), no model name
anywhere, so gen_spike_data.py keeps the thinking verbatim.

Usage:
  python3 gen_spike_extra.py --out spike-extra.txt --seed 11
"""

import argparse
import random


def _t(user, thinking, response):
    return (user, thinking, response)


# ---------------------------------------------------------------- math ---

def math_turns(r: random.Random, n=1000):
    out = []
    goals = [("add", 350), ("sub", 300), ("mul", 200), ("div", 150)]
    assert sum(w for _, w in goals) == n
    for op, want in goals:
        made = 0
        tries = 0
        seen_local = set()
        while made < want and tries < want * 500:
            tries += 1
            if op == "add":
                a, b = r.randint(10, 99), r.randint(10, 99)
                ans = a + b
                th = (f"add {a} and {b}. tens: {a // 10 * 10}+{b // 10 * 10} "
                      f"is {a // 10 * 10 + b // 10 * 10}. ones: {a % 10}+{b % 10} "
                      f"is {a % 10 + b % 10}. total {ans}.")
                cand = _t(f"what is {a} plus {b}", th, f"{ans}.")
            elif op == "sub":
                a = r.randint(20, 120)
                b = r.randint(10, a - 1)
                ans = a - b
                cand = _t(
                    f"what is {a} minus {b}",
                    f"subtract {b} from {a}. {a} take away {b} leaves {ans}.",
                    f"{ans}.")
            elif op == "mul":
                a, b = r.randint(2, 20), r.randint(2, 20)
                ans = a * b
                cand = _t(
                    f"what is {a} times {b}",
                    f"multiply {a} by {b}. {a} groups of {b} is {ans}.",
                    f"{ans}.")
            else:
                b = r.randint(2, 20)
                ans = r.randint(2, 20)
                c = b * ans
                cand = _t(
                    f"what is {c} divided by {b}",
                    f"divide {c} by {b}. {b} goes into {c} {ans} times.",
                    f"{ans}.")
            if cand not in seen_local:
                seen_local.add(cand)
                out.append(cand)
                made += 1
    return out


# ------------------------------------------------------------ reasoning ---

SYL = [
    (("cats", "cat"), ("animals", "animal")),
    (("dogs", "dog"), ("animals", "animal")),
    (("sparrows", "sparrow"), ("birds", "bird")),
    (("eagles", "eagle"), ("birds", "bird")),
    (("salmon", "salmon"), ("fish", "fish")),
    (("sharks", "shark"), ("fish", "fish")),
    (("roses", "rose"), ("flowers", "flower")),
    (("tulips", "tulip"), ("flowers", "flower")),
    (("oaks", "oak"), ("trees", "tree")),
    (("ants", "ant"), ("insects", "insect")),
    (("hammers", "hammer"), ("tools", "tool")),
    (("cars", "car"), ("machines", "machine")),
    (("trucks", "truck"), ("vehicles", "vehicle")),
    (("apples", "apple"), ("fruits", "fruit")),
    (("pianos", "piano"), ("instruments", "instrument")),
    (("crows", "crow"), ("birds", "bird")),
    (("trout", "trout"), ("fish", "fish")),
    (("daisies", "daisy"), ("flowers", "flower")),
    (("pines", "pine"), ("trees", "tree")),
    (("bees", "bee"), ("insects", "insect")),
    (("saws", "saw"), ("tools", "tool")),
    (("buses", "bus"), ("vehicles", "vehicle")),
    (("oranges", "orange"), ("fruits", "fruit")),
]

NAMES = ["felix", "max", "bella", "rocky", "luna", "milo", "oscar",
         "ruby", "leo", "mia", "toby", "zoe", "jack", "ivy", "sam",
         "nora", "finn", "ada", "pip", "gus", "hal", "june", "otto",
         "vera", "wren", "yara", "zane", "cleo", "rex", "skye"]

PONENS = [
    ("it rains", "the ground gets wet"),
    ("you heat ice", "it melts"),
    ("a plant gets no water", "it wilts"),
    ("you drop glass", "it breaks"),
    ("it is night", "the sky is dark"),
    ("you mix red and blue", "you get purple"),
    ("metal gets hot", "it expands"),
    ("you skip sleep", "you feel tired"),
    ("water reaches 100 celsius", "it boils"),
    ("you press the switch", "the light turns on"),
    ("it snows", "the road gets slippery"),
    ("you add salt to food", "it tastes salty"),
    ("you run fast", "your heart beats faster"),
    ("soap touches grease", "the grease loosens"),
    ("you water a seed", "it sprouts"),
    ("iron gets wet", "it rusts"),
    ("you shout in a cave", "you hear an echo"),
    ("a baby is hungry", "it cries"),
    ("you open the oven", "heat escapes"),
    ("birds migrate south", "winter is coming"),
    ("you freeze water", "it becomes ice"),
    ("a dog sees food", "it wags its tail"),
    ("you study hard", "you learn more"),
    ("plants get sunlight", "they grow taller"),
    ("you boil an egg", "it hardens"),
    ("wind blows hard", "trees sway"),
    ("you eat too much", "your stomach hurts"),
    ("a car runs out of fuel", "it stops moving"),
    ("you touch fire", "it burns you"),
    ("birds build nests", "eggs stay safe"),
    ("you save money", "it adds up"),
    ("rain falls all week", "rivers flood"),
    ("you skip brushing", "teeth decay"),
    ("a phone charges", "its battery fills"),
    ("snow piles up", "schools close"),
]

TOLLENS = [
    ("it rains", "the ground is wet", "the ground is dry"),
    ("you heat ice", "it melts", "the ice is solid"),
    ("a plant gets no water", "it wilts", "the plant stands tall"),
    ("you drop glass", "it breaks", "the glass is whole"),
    ("it is night", "the sky is dark", "the sky is bright"),
    ("you press the switch", "the light turns on", "the light is off"),
    ("it snows", "the road is slippery", "the road is dry"),
    ("iron gets wet", "it rusts", "the iron shines clean"),
    ("you shout in a cave", "you hear an echo", "all is silent"),
    ("a baby is hungry", "it cries", "the baby sleeps soundly"),
    ("you freeze water", "it becomes ice", "the water still flows"),
    ("you boil an egg", "it hardens", "the egg stays runny"),
    ("a dog sees food", "it wags its tail", "its tail hangs still"),
    ("plants get sunlight", "they grow taller", "the plants stay short"),
    ("wind blows hard", "trees sway", "the trees stand still"),
    ("you run fast", "your heart beats faster", "your pulse stays slow"),
    ("soap touches grease", "the grease loosens", "the grease stays stuck"),
    ("you water a seed", "it sprouts", "the soil stays bare"),
    ("you open the oven", "heat escapes", "the kitchen stays cool"),
    ("you study hard", "you learn more", "you remember nothing"),
    ("a fire burns", "smoke rises", "the air stays clear"),
    ("winter comes", "lakes freeze", "the lake stays liquid"),
    ("you stretch a rubber band", "it gets longer", "its length never changes"),
    ("a bell rings", "sound spreads", "all stays silent"),
    ("you plant corn", "stalks grow", "the field stays empty"),
    ("rain soaks the soil", "worms surface", "no worms appear"),
    ("you dim the lamp", "the room darkens", "the room stays bright"),
    ("a hammer hits a nail", "the nail sinks in", "the nail never moves"),
    ("you feed a fire", "it grows hotter", "the flames die down"),
]

OBJECTS = ["ball", "cup", "box", "door", "leaf", "pen", "hat", "key",
           "book", "chair", "table", "bottle"]
COLORS = ["red", "blue", "green", "yellow", "black", "white"]


def reasoning_turns(r: random.Random, n=1000):
    out = []
    goals = [("syl", 200), ("syl_inv", 150), ("ponens", 140),
             ("tollens", 90), ("either", 140), ("sort4", 150),
             ("order", 130)]
    assert sum(w for _, w in goals) == n
    for kind, want in goals:
        made, tries = 0, 0
        seen_local = set()
        while made < want and tries < want * 500:
            tries += 1
            cand = None
            if kind == "syl":
                (a, a1), (b, b1) = r.choice(SYL)
                name = r.choice(NAMES)
                cand = _t(
                    f"all {a} are {b}. {name} is a {a1}. is {name} a {b1}",
                    f"{name} is one of the {a}, and all {a} are {b}. so yes.",
                    "yes.")
            elif kind == "syl_inv":
                # Affirming the consequent is INVALID -- teach the boundary.
                (a, a1), (b, b1) = r.choice(SYL)
                name = r.choice(NAMES)
                cand = _t(
                    f"all {a} are {b}. {name} is a {b1}. is {name} a {a1}",
                    f"all {a} are {b}, but not every {b1} is a {a1}. "
                    f"{name} could be another {b1}. not necessarily.",
                    "not necessarily.")
            elif kind == "ponens":
                p, q = r.choice(PONENS)
                v = r.randint(1, 5)
                if v == 1:
                    cand = _t(
                        f"if {p}, {q}. {p}. {q}?",
                        f"the condition holds. if {p} then {q}, and {p}. so yes.",
                        "yes.")
                elif v == 2:
                    cand = _t(
                        f"if {p}, then {q}. {p}. so {q}?",
                        f"if {p} then {q}, and {p}. so yes.",
                        "yes.")
                elif v == 3:
                    cand = _t(
                        f"given that if {p}, {q}: {p} is true. is {q} true?",
                        f"if {p} then {q} is given, and {p} is true. so yes.",
                        "yes.")
                elif v == 4:
                    cand = _t(
                        f"whenever {p}, {q}. {p} now. {q}?",
                        f"{p} now, and whenever {p}, {q}. so yes.",
                        "yes.")
                else:
                    cand = _t(
                        f"since {p} implies {q}, and {p}: {q}?",
                        f"{p} implies {q}, and {p}. so yes.",
                        "yes.")
            elif kind == "tollens":
                p, q, notq = r.choice(TOLLENS)
                v = r.randint(1, 4)
                if v == 1:
                    cand = _t(
                        f"if {p}, {q}. but {notq}. did {p}",
                        f"{notq}, so {q} is not true. {p} did not happen. no.",
                        "no.")
                elif v == 2:
                    cand = _t(
                        f"given that if {p}, {q}: {notq}. did {p}?",
                        f"{notq} rules out {q}, so {p} did not happen. no.",
                        "no.")
                elif v == 3:
                    cand = _t(
                        f"{notq}, yet if {p} then {q}. could {p} have happened?",
                        f"{notq} means {q} is false, so {p} could not happen. no.",
                        "no.")
                else:
                    cand = _t(
                        f"suppose {p}. then {q} would hold. yet {notq}. "
                        f"was the supposition right?",
                        f"{notq} contradicts {q}, so the supposition fails. no.",
                        "no.")
            elif kind == "either":
                obj = r.choice(OBJECTS)
                x, y = r.sample(COLORS, 2)
                cand = _t(
                    f"the {obj} is either {x} or {y}. it is not {x}. "
                    f"which color is it",
                    f"it is not {x}, so it must be {y}.",
                    f"{y}.")
            elif kind == "sort4":
                xs = r.sample(range(1, 100), 4)
                s = sorted(xs)
                cand = _t(
                    f"sort {xs[0]}, {xs[1]}, {xs[2]}, {xs[3]} smallest first",
                    f"order them. {s[0]}, then {s[1]}, then {s[2]}, then {s[3]}.",
                    f"{s[0]}, {s[1]}, {s[2]}, {s[3]}.")
            else:  # order: transitive finish chains, huge name space
                a, b, c = r.sample(NAMES, 3)
                if r.random() < 0.5:
                    cand = _t(
                        f"{a} finished before {b}. {b} finished before {c}. "
                        f"who finished first?",
                        f"{a} beat {b} and {b} beat {c}, so {a} is first.",
                        f"{a}.")
                else:
                    cand = _t(
                        f"{a} finished before {b}. {b} finished before {c}. "
                        f"who finished last?",
                        f"{a} beat {b} and {b} beat {c}, so {c} is last.",
                        f"{c}.")
            if cand is not None and cand not in seen_local:
                seen_local.add(cand)
                out.append(cand)
                made += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="spike-extra.txt")
    p.add_argument("--math", type=int, default=1000)
    p.add_argument("--reason", type=int, default=1000)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    r = random.Random(args.seed)
    turns = math_turns(r, args.math) + reasoning_turns(r, args.reason)
    r.shuffle(turns)

    seen, uniq = set(), []
    for t in turns:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(
            f"User: {u}\nThinking: {t}\nResponse: {r}"
            for u, t, r in uniq) + "\n")
    print(f"[gen] wrote {args.out} ({len(uniq)} turns, "
          f"{len(turns) - len(uniq)} dupes dropped)")


if __name__ == "__main__":
    main()
