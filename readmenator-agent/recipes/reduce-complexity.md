# Recipe: Reduce File Complexity

Target hotspot: `tests/test_integration.py`
(complexity 1.0, centrality 0.9)

1. Read dependents: `grep -n 'tests/test_integration.py' readmenator-agent/ARCHITECTURE.md`
2. Extract functions/classes into new files in the same subsystem
3. Update imports
4. Regenerate: `readmenator .`
