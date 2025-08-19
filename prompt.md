No reasoning model prompt:

```python
f'''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model. Generate only the JSON file, no other text.
</objective>

<instruction>
You will be given a natural language description of a CAD design task. Your goal is to convert it into a structured JSON representation, which includes sketch geometry and extrusion operations.
The extrusion <operation> must be one of the following:
1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
Ensure all coordinates, geometry, and extrusion depths are extracted accurately from the input.
</instruction>

<description>
{prompt}
</description>'''
```

Reasoning model prompt:

```python
f'''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model based on the description provided. The output should include a reasoning section within the <think> tag and the corresponding JSON in the <json> tag. Do not provide any additional text outside of the tags.
</objective>

<instruction>
You will be given a natural language description of a CAD design task enclosed within <description> </description>. Your task is to:
1. Analyze the description and extract the relevant geometric and extrusion information.
2. In the <think> tag, explain how you derived each field and value in the JSON from the description. This includes the geometric properties (e.g., coordinates, shapes) and extrusion operations. The reasoning should clarify how the geometry is mapped to the JSON structure and the chosen extrusion operation.
3. Based on the reasoning in the <think> tag, generate the corresponding JSON structure for the CAD model in the <json> tag.

The extrusion <operation> must be one of the following:
1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
</instruction>

<description>
{prompt}
</description>'''
```

Multi-turn reasoning prompt:

```python
f'''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model based on the description provided. The output should include a reasoning section within the <think> tag and the corresponding JSON in the <json> tag. Do not provide any additional text outside of the tags.
</objective>

<instruction>
You will be given a natural language description of a CAD design task enclosed within the <description> tag. Additionally, you may also receive up to two previous turns of input/output context in <previous_turn_1> and <previous_turn_2>. These turns represent earlier steps in a multi-step CAD design process.

Each previous turn includes both the <think> and <json> tags, providing the model's reasoning and structured output from earlier steps.

- If the tags <previous_turn_1> or <previous_turn_2> are empty, it means there is no prior context for that slot.
- Use any available previous turns to inform geometric references, operation chaining, or any dependencies needed in the current modeling step.

Your task is to:

1. Analyze the current <description> and extract relevant geometric and extrusion information.
2. Reference earlier steps from <previous_turn_1> and <previous_turn_2> if available to maintain continuity and accuracy.
3. In the <think> tag, explain how you derived each field and value in the JSON, including reasoning for geometry, dependencies on earlier steps, and chosen operations.
4. In the <json> tag, return only the structured CAD instruction in JSON format corresponding to this design step.

The extrusion <operation> must be one of the following:

1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
</instruction>

<previous_turn_1>
</previous_turn_1>

<previous_turn_2>
</previous_turn_2>

<description>
{prompt}
</description>'''
```