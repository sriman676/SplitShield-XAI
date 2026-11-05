# 💬 The assistant (ShieldLM)

[← Back to the main README](../README.md)

Ask questions in plain English on the dashboard's **💬 Ask SplitShield** page.

## 💬 What it answers

Answers full-sentence questions from the knowledge base and your logs. The language
model only words incident narratives, and any narrative that fails the faithfulness
check is replaced by a template.

- 📖 *"What is a brute force attack?"* gives the types, how it is carried out step by
  step, signs in your logs, defences, the ATT&CK technique, and whether it is in your
  logs now.
- 🔎 *"Is there any defense evasion happening?"* checks every ingested log and answers
  🚨 `ONGOING` (activity in the last hour), 🕒 `HAPPENED EARLIER` (first and last
  seen), ⚠️ `POSSIBLE` (low-confidence only) or ✅ `NOT SEEN` (with the logs checked).
  When the evidence is thin, it suggests follow-up questions.
- 🧠 Remembers the topic: *"is it still going on?"*, *"how do I stop it?"*.
- 🧾 *"Explain T1059"* or *"explain incident 3"* gives a five-part write-up.

## 📈 How well it works

| | |
|---|---|
| 💬 Question understanding | **87.6 %** on question phrasings never seen in training (grouped CV 88.0 % ± 1.9) |
| 🧠 ShieldLM | trained from scratch, 1.95 M parameters, held-out perplexity 1.21 |
| 🎯 RLCD alignment | faithfulness 0.924 → **0.998**, win rate 0.80 |
| 🗜️ Compression | pruned and distilled to **20 %** of the size (faithfulness 0.999, perplexity +1.2 %) |

## 🗣️ How it answers

- **Answer first:** the first line is the plain answer (*Yes, brute force is ongoing
  from 203.0.113.45*), then evidence, then details.
- **What to do now:** numbered steps (contain, investigate, fix, verify) for anything
  found in your logs. Blocking stays your decision.
- **Adapts to the situation:** an active attack gets a short answer and the steps
  first; *what is …* gets a full explanation; expert wording (technique IDs, event
  IDs, *query*) gets IDs, fields and example queries. You can force Beginner or
  Expert on the Ask page.
- **Typos are fine:** *whats brut force* is understood, with a *Did you mean …?*
  line when unsure (90.7 % of questions with typos understood).
- **Next questions:** 2-3 clickable follow-ups after every answer.
- **Searches and reports:** *show failed logins for bob yesterday*, *make a report
  of this week* (see [SOC](SOC.md#-search-and--reports)).

## 👍 It learns from every question

Under each reply there is a 👍 / 👎. A 👍 confirms how the question was understood; a
👎 lets you pick what you were actually asking for. Each rating immediately retrains
the question-understanding model in the background (a few seconds), and the new model
is kept **only if**:

1. its accuracy on fixed held-out question templates (never trained on) does not drop, and
2. it understands at least as many of the rated questions as the current model.

Otherwise the current model stays. The previous model is kept for rollback. Ratings are
stored in `artifacts/nlp/question_feedback.jsonl`.

What this changes is how a question is *understood*, never what is *claimed*: facts in
answers always come from the knowledge base and your logs. Big chat assistants such as
ChatGPT or Claude do not retrain live on each chat either; they collect feedback and
improve in gated training rounds, which is what happens here on a small scale. The
knowledge itself (ATT&CK, Sigma rules, threat intel) stays current through the
[update agent](UPDATE_AGENT.md) when it is switched on.

ShieldLM, the narrative writer, improves once a night from 👍 / 👎 on incident narratives
(Incidents page), under the same keep-only-if-not-worse rule plus a speed check.

