# Examples

Two programs, each a whole use of Ada from an empty model to a function your software calls. Each has a step-by-step guide in the docs.

| Example | Language | Guide |
|---|---|---|
| [Route support tickets](support-tickets/tickets.py) | Python | [docs.glyphh.ai/ada-guide-tickets](https://docs.glyphh.ai/ada-guide-tickets) |
| [Customer churn](churn/churn.ts) | TypeScript | [docs.glyphh.ai/ada-guide-churn](https://docs.glyphh.ai/ada-guide-churn) |

Both create a model, so run them with an org admin's API key in `GLYPHH_API_KEY`. Every call is metered on your organization's plan: the tickets example makes about 115 calls, the churn example about 200.

```bash
cd support-tickets && pip install glyphh-ada && python tickets.py
cd churn && npm install && npm start
```
