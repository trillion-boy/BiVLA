"""Regenerate every catalogued figure from the original experiment records."""
import json
import common
if __name__ == '__main__':
    for spec in json.loads((common.OUT/'figure_manifest.json').read_text()):
        getattr(common,spec['function'])(spec['name'],*spec['args'])
        print(spec['name'],flush=True)
