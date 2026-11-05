# 📖 Glossary

[← Back to the main README](../README.md)

Terms used across the documentation.

## 📖 Glossary

| Term | Meaning |
|---|---|
| Macro-F1 | Average of per-class F1 scores (0–1); every attack type counts equally |
| Train / validation / test | Learn on train, choose settings on validation, report test once |
| Held-out | Data never used to build or tune what is being scored |
| Blind test | A whole extra dataset scored once at the very end, never looked at before |
| Over- / underfitting | Much better on train than on new data / poor on both |
| False-alarm rate | Share of normal events or flows that raised an alert |
| ATT&CK technique / tactic | MITRE's catalogue of *how* attackers act (technique) and *why* (tactic) |
| Sigma | Open format and community library of detection rules |
| PSI | Population Stability Index: how much live traffic differs from training data |
| Champion / challenger | A new model replaces the current one only if it is not worse on fixed held-out data |
| Alarm budget | The most false alarms per day an operator accepts; the threshold is set to stay within it |
