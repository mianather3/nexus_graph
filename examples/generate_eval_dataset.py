#!/usr/bin/env python3
"""Generates examples/eval_dataset.json: a SYNTHETIC multi-hop retrieval benchmark.

    python3 examples/generate_eval_dataset.py > examples/eval_dataset.json

The output is deterministic (fixed seed, stdlib only).

What is simulated
-----------------
A fictional engineering organisation: teams own services, services depend on
datastores (and sometimes on other teams' services), datastores sit on shared
infrastructure zones, and teams produce container images.

There is no real embedding model behind the vectors. Each entity's embedding is
its domain's random "topic" direction plus a per-type direction plus noise,
which mimics the one property the evaluation relies on: a dense encoder places
things about the same subject close together, regardless of how they are
connected.

Three query classes, each built to isolate one retrieval behaviour:

* lexical    - the query names an opaque identifier (a datastore code such as
               "q7f3"). The text matches exactly one entity; the embedding only
               says "some datastore".
* semantic   - the query paraphrases a team's remit with words that appear in
               no entity description. The embedding points at the right domain;
               the text matches nothing useful.
* multi_hop  - the query names a service and asks for something reachable only
               through the graph (its datastore, or the infrastructure zone
               under that datastore). The answer shares neither vocabulary nor
               topic with the query.

Because the classes are constructed this way, the results show HOW the systems
differ, not HOW MUCH a real deployment would gain.
"""

import json
import math
import random
import sys

SEED = 20240611
DIM = 32
NOISE = 0.30

# name: (description keywords, paraphrase words used ONLY in semantic queries)
DOMAINS = {
    "payments": (["payment", "invoice", "refund", "chargeback"], ["money", "collecting", "customers", "owed"]),
    "identity": (["login", "password", "session", "credential"], ["proving", "who", "someone", "claims"]),
    "search": (["query", "ranking", "indexing", "autocomplete"], ["finding", "things", "typed", "words"]),
    "catalog": (["product", "listing", "category", "inventory"], ["goods", "offered", "shelves", "assortment"]),
    "shipping": (["parcel", "courier", "tracking", "delivery"], ["getting", "packages", "doorstep", "transit"]),
    "messaging": (["email", "notification", "template", "campaign"], ["reaching", "people", "inbox", "announcements"]),
    "analytics": (["metric", "dashboard", "funnel", "cohort"], ["understanding", "behaviour", "numbers", "trends"]),
    "media": (["video", "thumbnail", "transcode", "playback"], ["watching", "clips", "footage", "streams"]),
    "fraud": (["risk", "anomaly", "blocklist", "scoring"], ["catching", "cheaters", "suspicious", "abuse"]),
    "support": (["ticket", "agent", "escalation", "helpdesk"], ["assisting", "unhappy", "complaints", "resolving"]),
    "pricing": (["discount", "tariff", "coupon", "quote"], ["deciding", "cost", "cheaper", "promotions"]),
    "recommend": (["suggestion", "personalization", "affinity", "feed"], ["guessing", "taste", "liked", "similar"]),
    "ledger": (["accounting", "journal", "reconciliation", "balance"], ["bookkeeping", "tallies", "audit", "totals"]),
    "maps": (["geocode", "route", "tile", "address"], ["navigating", "places", "directions", "nearby"]),
    "reviews": (["rating", "comment", "moderation", "feedback"], ["opinions", "stars", "testimonials", "voices"]),
    "booking": (["reservation", "calendar", "availability", "slot"], ["scheduling", "appointments", "arranging", "visits"]),
    "partner": (["integration", "webhook", "marketplace", "onboarding"], ["outside", "companies", "connecting", "third"]),
    "privacy": (["consent", "erasure", "retention", "compliance"], ["respecting", "personal", "rights", "regulators"]),
    "devtools": (["pipeline", "artifact", "release", "rollback"], ["shipping", "code", "engineers", "faster"]),
    "storage": (["upload", "archive", "attachment", "quota"], ["keeping", "files", "documents", "safe"]),
}
ROLES = ["api", "worker", "gateway"]
ZONES = 8


def unit(v):
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def gauss(rng):
    return [rng.gauss(0.0, 1.0) for _ in range(DIM)]


def blend(rng, parts, noise=NOISE):
    """Unit vector: weighted sum of directions plus isotropic noise."""
    out = [0.0] * DIM
    for weight, direction in parts:
        for i, x in enumerate(direction):
            out[i] += weight * x
    jitter = unit(gauss(rng))
    return [round(x, 4) for x in unit([o + noise * j for o, j in zip(out, jitter)])]


def code(rng, used):
    while True:
        value = rng.choice("ghjkmnpqrstvwxz") + "".join(rng.choice("0123456789abcdef") for _ in range(3))
        if value not in used:
            used.add(value)
            return value


def main():
    rng = random.Random(SEED)
    topic = {name: unit(gauss(rng)) for name in DOMAINS}
    kind = {name: unit(gauss(rng)) for name in ["team", "service", "datastore", "zone", "image"]}
    used_codes = set()

    entities, relations = [], []

    def add(name, etype, description, embedding):
        entities.append({
            "name": name,
            "type": etype,
            "attributes": {"description": description},
            "embedding": embedding,
        })

    def link(source, target, relation, weight):
        relations.append({"source": source, "target": target, "relation": relation, "weight": weight})

    zones = []
    for z in range(ZONES):
        zone_code = code(rng, used_codes)
        name = f"zone-{zone_code}"
        zones.append(name)
        add(name, "Service", f"Infrastructure zone {zone_code} providing compute racks and block volumes",
            blend(rng, [(1.0, kind["zone"])]))

    services, datastores, datastore_zone, service_stores, team_of = {}, {}, {}, {}, {}
    for domain, (words, _) in DOMAINS.items():
        team = f"{domain}-team"
        add(team, "Organization", f"Team responsible for {words[0]} and {words[1]} systems",
            blend(rng, [(1.0, topic[domain]), (0.35, kind["team"])]))

        stores = []
        for i in range(2):
            store_code = code(rng, used_codes)
            store = f"{domain}-db-{store_code}"
            stores.append(store)
            zone = rng.choice(zones)
            datastore_zone[store] = zone
            add(store, "Service", f"Datastore cluster {store_code} persisting {words[i + 2]} records",
                blend(rng, [(0.55, topic[domain]), (0.85, kind["datastore"])]))
            link(store, zone, "DependsOn", round(rng.uniform(0.8, 1.0), 2))
        datastores[domain] = stores

        names = []
        for i, role in enumerate(ROLES):
            service = f"{domain}-{role}-svc"
            names.append(service)
            team_of[service] = team
            add(service, "Service", f"The {role} tier serving {words[i]} and {words[i + 1]} requests",
                blend(rng, [(1.0, topic[domain]), (0.35, kind["service"])]))
            link(team, service, "Contains", 1.0)
            store = stores[i % 2]
            service_stores[service] = [store]
            link(service, store, "DependsOn", round(rng.uniform(0.7, 1.0), 2))
        services[domain] = names

        for i in range(2):
            image_code = code(rng, used_codes)
            image = f"image-{image_code}"
            add(image, "Artifact", f"Container image build {image_code} for the {domain} {ROLES[i]} tier",
                blend(rng, [(0.5, topic[domain]), (0.8, kind["image"])]))
            link(image, team, "ProducedBy", 1.0)

    # Cross-domain service dependencies, so that neighbourhoods are not tidy stars.
    domain_names = list(DOMAINS)
    for domain in domain_names:
        for service in services[domain]:
            if rng.random() < 0.35:
                other = rng.choice([d for d in domain_names if d != domain])
                link(service, rng.choice(services[other]), "DependsOn", round(rng.uniform(0.5, 0.9), 2))

    queries = []

    # lexical: opaque identifier lookups.
    lexical_targets = [store for stores in datastores.values() for store in stores]
    rng.shuffle(lexical_targets)
    for n, store in enumerate(lexical_targets[:20]):
        store_code = store.rsplit("-", 1)[1]
        queries.append({
            "id": f"lexical-{n:02d}",
            "category": "lexical",
            "text": f"Which cluster has the identifier {store_code}?",
            # The encoder only understands "some datastore".
            "embedding": blend(rng, [(1.0, kind["datastore"])]),
            "relevant": [store],
            "evidence_path": [],
        })

    # semantic: paraphrases sharing no vocabulary with any description.
    for n, (domain, (_, paraphrase)) in enumerate(DOMAINS.items()):
        queries.append({
            "id": f"semantic-{n:02d}",
            "category": "semantic",
            "text": f"Who looks after {paraphrase[0]} {paraphrase[1]} {paraphrase[2]} {paraphrase[3]}?",
            "embedding": blend(rng, [(1.0, topic[domain]), (0.35, kind["team"])]),
            "relevant": [f"{domain}-team"],
            "evidence_path": [],
        })

    # multi_hop: the answer is reachable only through the graph.
    hop_services = [service for names in services.values() for service in names]
    rng.shuffle(hop_services)
    for n, service in enumerate(hop_services[:30]):
        domain, role, _ = service.split("-")
        store = service_stores[service][0]
        zone = datastore_zone[store]
        query_embedding = blend(rng, [(1.0, topic[domain]), (0.35, kind["service"])])
        if n % 2 == 0:
            queries.append({
                "id": f"multihop-{n:02d}",
                "category": "multi_hop_1",
                "text": f"Where does the {domain} {role} svc keep its state?",
                "embedding": query_embedding,
                "relevant": [store],
                "evidence_path": [service, store],
            })
        else:
            queries.append({
                "id": f"multihop-{n:02d}",
                "category": "multi_hop_2",
                "text": f"Which hardware ultimately sits underneath the {domain} {role} svc?",
                "embedding": query_embedding,
                "relevant": [zone],
                "evidence_path": [service, store, zone],
            })

    dataset = {
        "name": "synthetic-org-v1",
        "description": (
            "Synthetic engineering-organisation graph with simulated embeddings. Generated by "
            "examples/generate_eval_dataset.py; illustrates how the retrieval modes differ, "
            "and is not evidence of real-world accuracy."
        ),
        "graph": {"entities": entities, "relations": relations},
        "queries": queries,
    }
    json.dump(dataset, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
