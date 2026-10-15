"""Snapshot of baseline primitives; H1 changes the retrieval vectors only."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from .io import fingerprint

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
GUARD_RE = re.compile(r"\b(?:if|while|for)\s*\(([^\n{}]+)\)")

def features(code: str) -> dict[str, set[str]]:
    return {
        "tokens": {item.lower() for item in TOKEN_RE.findall(code)},
        "calls": {item.lower() for item in CALL_RE.findall(code)},
        "guards": {" ".join(item.split()).lower() for item in GUARD_RE.findall(code)},
    }



def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)



def fit_prompt(template: str, target_code: str, graph_context: str, demos: list[dict[str, Any]], instruction: str, policy: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    old_instruction = "Determine whether the target function contains a software vulnerability.\nUse the target code, graph context, and labeled reference examples as evidence."
    if old_instruction not in template:
        raise ValueError("Contract prompt template changed; expected instruction anchor is missing")
    template = template.replace(old_instruction, instruction)
    target_cap = int(policy["target_code_max_chars"])
    demo_cap = int(policy["standard_demo_max_chars"])
    target = target_code[:target_cap]
    graph = graph_context
    demo_texts = [("Vulnerable" if int(demo["label"]) == 1 else "Non-vulnerable", demo["code"][:demo_cap]) for demo in demos]
    def render(t: str, g: str, ds: list[tuple[str, str]]) -> str:
        demonstration_text = "\n\n".join(f"{label} example:\n{code}" for label, code in ds)
        return template.replace("{target_code}", t).replace("{graph_context}", g).replace("{demonstrations}", demonstration_text)
    prompt = render(target, graph, demo_texts)
    while len(prompt) > int(policy["context_limit"]) and any(code for _, code in demo_texts):
        index = max(range(len(demo_texts)), key=lambda i: len(demo_texts[i][1]))
        label, code = demo_texts[index]
        reduction = min(len(code), max(1, len(prompt) - int(policy["context_limit"])))
        demo_texts[index] = (label, code[: len(code) - reduction])
        prompt = render(target, graph, demo_texts)
    if len(prompt) > int(policy["context_limit"]):
        reduction = min(len(graph), max(0, len(prompt) - int(policy["context_limit"])))
        graph = graph[: len(graph) - reduction]
        prompt = render(target, graph, demo_texts)
    if len(prompt) > int(policy["context_limit"]):
        reduction = min(len(target), max(0, len(prompt) - int(policy["context_limit"])))
        target = target[: len(target) - reduction]
        prompt = render(target, graph, demo_texts)
    if len(prompt) > int(policy["context_limit"]):
        raise ValueError(f"Prompt exceeds contract context limit: {len(prompt)}")
    return prompt, {
        "context_limit": int(policy["context_limit"]), "prompt_chars": len(prompt),
        "target_chars_used": len(target), "graph_chars_used": len(graph),
        "demo_chars_used": [len(code) for _, code in demo_texts],
    }



def parse_graphml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g":"http://graphml.graphdrawing.org/xmlns"}
    keys = {k.attrib["id"]:k.attrib.get("attr.name",k.attrib["id"]) for k in root.findall("g:key",ns)}
    def properties(element):
        return {keys[d.attrib["key"]]:d.text or "" for d in element.findall("g:data",ns)}
    nodes=[]
    for node in root.findall(".//g:node",ns):
        props=properties(node)
        kind=props.get("labelV",props.get("_label",props.get("label","")))
        nodes.append({"id":node.attrib["id"],"type":kind,"code":props.get("CODE",props.get("code","")),"name":props.get("NAME",props.get("name","")),"external":props.get("IS_EXTERNAL",props.get("isExternal","false")).lower()=="true","order":int(props.get("ORDER",props.get("order","0")) or 0)})
    edges=[]
    for edge in root.findall(".//g:edge",ns):
        props=properties(edge)
        edges.append({"src":edge.attrib["source"],"dst":edge.attrib["target"],"label":props.get("labelE",props.get("_label",props.get("label","")))})
    if not nodes or not any(n["type"] for n in nodes):
        raise ValueError("Whole CPG GraphML has no typed nodes")
    by_id={n["id"]:n for n in nodes}
    ast_edges=[e for e in edges if e["label"]=="AST"]
    children={}
    parented=set()
    for edge in ast_edges:
        if edge["src"] not in by_id or edge["dst"] not in by_id:
            raise ValueError("Dangling AST edge")
        children.setdefault(edge["src"],[]).append(edge["dst"])
        parented.add(edge["dst"])
    roots=[n["id"] for n in nodes if n["type"]=="METHOD" and not n["external"] and n["name"]!="<global>" and n["id"] in children]
    # Whole source-function CPG: remove external method stubs/global metadata,
    # not source statements. This is ownership filtering, not a sink slice.
    owned=set()
    def ownership(nid):
        if nid in owned:
            return
        owned.add(nid)
        for child in children.get(nid,[]):
            ownership(child)
    for nid in roots:
        ownership(nid)
    full_node_count,full_edge_count=len(nodes),len(edges)
    nodes=[n for n in nodes if n["id"] in owned]
    edges=[e for e in edges if e["src"] in owned and e["dst"] in owned]
    ast_edges=[e for e in edges if e["label"]=="AST"]
    tokens=[]
    active=set()
    def visit(nid):
        if nid in active:
            raise ValueError("AST cycle")
        active.add(nid)
        kind=by_id[nid]["type"]
        # Sim_SBT uses simplified PREORDER, unlike SBT's repeated type labels
        # and parentheses (ComFormer §III-A.2, Fig. 2; cited by GRACE).
        tokens.append(kind)
        for child in sorted(children.get(nid,[]),key=lambda x:(by_id[x]["order"],x)):
            visit(child)
        active.remove(nid)
    for nid in sorted(roots):
        visit(nid)
    # This reproduces genexample.py's sequence primitive on AST edge text;
    # upstream train_ast_clean.csv and its serializer are not shipped.
    edge_tokens=" ".join(f"{e['src']}->{e['dst']}[{e['label']}]" for e in ast_edges).split()
    return {"nodes":nodes,"edges":edges,"simsbt":tokens,"ast_edge_tokens":edge_tokens,"edge_types":dict(Counter(e["label"] for e in edges)),"simsbt_method":"ordered_joern_ast_type_preorder_v2","graph_status":"joern_exported" if nodes and edges else "joern_no_function_ast","exported_node_count":full_node_count,"exported_edge_count":full_edge_count}



def sequence_similarity(a: list[str], b: list[str], substitution_cost: int) -> float:
    if not a and not b:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein
        distance=Levenshtein.distance(a,b,weights=(1,1,substitution_cost))
    except ImportError:
        previous=list(range(len(b)+1))
        for i,left in enumerate(a,1):
            current=[i]
            for j,right in enumerate(b,1):
                current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(0 if left==right else substitution_cost)))
            previous=current
        distance=previous[-1]
    return (len(a)+len(b)-distance)/(len(a)+len(b))



def jaccard_tokens(a: str,b: str) -> float:
    left,right=set(a.split()),set(b.split())
    return len(left&right)/len(left|right) if left|right else 1.0



def raw_graph_context(graph: dict,policy: dict) -> tuple[str,dict]:
    nodes=graph["nodes"][:int(policy["graph_nodes_max"])]
    if graph.get("schema") == "joern_legacy_text_v1":
        edges=graph["edges"][:int(policy["graph_edges_max"])]
        if not nodes or not graph["edges"]:
            return "GRAPH UNAVAILABLE: legacy Joern cache did not preserve both node and edge records.",{"nodes":0,"edges":0,"source_nodes":len(graph["nodes"]),"source_edges":len(graph["edges"]),"legacy_cache":True}
        node_lines=[]
        for i,n in enumerate(nodes):
            line=f"AST {i+1} | {n['type']}"
            if n.get("code"):
                line += " | " + n["code"]
            node_lines.append(line)
        text=("LEGACY JOERN GRAPH PREFIX (node IDs and node-to-edge mapping were not retained by the source cache):\n"
              + "\n".join(node_lines))
        text+="\nEDGES (original Joern endpoint IDs):\n"+"\n".join(f"{e['src']} -> {e['dst']} [{e['label']}]" for e in edges)
        return text,{"nodes":len(nodes),"edges":len(edges),"source_nodes":len(graph["nodes"]),"source_edges":len(graph["edges"]),"legacy_cache":True,"node_ids_available":False}
    selected={n["id"] for n in nodes}
    edges=[e for e in graph["edges"] if e["src"] in selected and e["dst"] in selected][:int(policy["graph_edges_max"])]
    if not graph["nodes"] or not graph["edges"]:
        return "GRAPH UNAVAILABLE: Joern could not extract a function AST/CPG from this exact snippet.",{"nodes":0,"edges":0,"source_nodes":len(graph["nodes"]),"source_edges":len(graph["edges"])}
    text="NODES (whole function CPG prefix; no sink selection):\n"+"\n".join(f"{n['id']} | {n['type']} | {n['code']}" for n in nodes)
    text+="\nEDGES:\n"+"\n".join(f"{e['src']} -> {e['dst']} [{e['label']}]" for e in edges)
    return text,{"nodes":len(nodes),"edges":len(edges),"source_nodes":len(graph["nodes"]),"source_edges":len(graph["edges"])}



def fit_contracted_prompt(template,target,graph,demos,instruction,policy):
    """Keep Full's prefix policy, including a safe bound for UTF-8 BPE text.

    The shared Full fitter interprets context_limit as characters. ASCII chars
    are a conservative token bound; for Unicode use UTF-8 bytes instead of
    silently allowing extra bytes/tokens. Record both units explicitly.
    """
    effective=dict(policy)
    cap=int(policy["context_limit"])
    while True:
        prompt,metadata=fit_prompt(template,target,graph,demos,instruction,effective)
        encoded=len(prompt.encode("utf-8"))
        if encoded<=cap:
            metadata.update(context_limit=cap,effective_character_budget=effective["context_limit"],prompt_utf8_bytes=encoded,token_count_measured=False)
            return prompt,metadata
        effective["context_limit"]-=max(1,encoded-cap)
        if effective["context_limit"]<1:
            raise ValueError("Contract prompt instructions exceed context budget")
