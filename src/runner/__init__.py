"""
Runner for the lifelong-learning benchmark.

Current implementation covers the minimal end-to-end loop (DBBench + zero_shot + single_agent):
- Reads experiment config from configs/assignment/default.yaml
- Uses scheduler to generate (task, index) execution schedule
- Calls backend /start_sample to get initial messages + tools
- Generates enhanced_messages via the zero_shot memory mechanism (currently pass-through)
- Executes samples using the single_agent execution engine (placeholder implementation)
- Saves history + result to disk for later use by memory mechanisms or analysis

Future additions:
- Real LLM calls (based on configs/llmapi/*.yaml)
- /interact interaction loop
- Other memory / execution mechanisms
"""
