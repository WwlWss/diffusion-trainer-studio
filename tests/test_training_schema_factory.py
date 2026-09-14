import json
import shutil
import subprocess
import unittest

from mikazuki.training_schema_factory import fixed_flux_family_schema, fixed_sd_schema


SCHEMA_STUB = r'''
function decorate(node) {
  node.meta = node.meta || {};
  for (const name of ["default", "description", "required", "disabled", "hidden", "min", "max", "step", "role"]) {
    node[name] = function(value = true) { this.meta[name] = value; return this; };
  }
  return node;
}
const Schema = {
  object(dict) { return decorate({type: "object", dict: dict || {}}); },
  intersect(list) { return decorate({type: "intersect", list}); },
  union(list) {
    return decorate({type: "union", list: list.map((item) =>
      (typeof item === "string" || typeof item === "number" || typeof item === "boolean") ? Schema.const(item) : item
    )});
  },
  const(value) { return decorate({type: "const", value}); },
  string() { return decorate({type: "string"}); },
  number() { return decorate({type: "number"}); },
  boolean() { return decorate({type: "boolean"}); },
  array(inner) { return decorate({type: "array", inner}); },
};

function collectKeys(schema, out = []) {
  if (!schema) return out;
  if (schema.type === "object" && schema.dict) {
    out.push(...Object.keys(schema.dict));
    for (const value of Object.values(schema.dict)) collectKeys(value, out);
  }
  if ((schema.type === "intersect" || schema.type === "union" || schema.type === "tuple") && schema.list) {
    for (const child of schema.list) collectKeys(child, out);
  }
  if (schema.inner) collectKeys(schema.inner, out);
  return out;
}
'''


FLUX_FAMILY_SYNTHETIC = r'''
Schema.intersect([
  Schema.object({
    model_type: Schema.union(["flux", "chroma", "anima"]).default("flux"),
    pretrained_model_name_or_path: Schema.string(),
  }),
  Schema.union([
    Schema.object({
      model_type: Schema.union(["flux", "chroma"]).required(),
      ae: Schema.string(),
      clip_l: Schema.string(),
      t5xxl: Schema.string(),
    }),
    Schema.object({
      model_type: Schema.const("anima").required(),
      anima_model_variant: Schema.union(["base", "2.9b"]).default("base"),
      qwen3: Schema.string(),
      vae: Schema.string(),
    }),
    Schema.object({}),
  ]),
  Schema.union([
    Schema.object({
      model_type: Schema.const("anima").required(),
      anima_training_mode: Schema.const("lora").required(),
      network_dim: Schema.number(),
    }),
    Schema.object({
      model_type: Schema.const("anima").required(),
      anima_training_mode: Schema.const("finetune").required(),
      self_attn_lr: Schema.number(),
      train_qwen3_text_encoder: Schema.boolean(),
    }),
    Schema.object({}),
  ]),
  Schema.union([
    Schema.object({
      model_type: Schema.union(["flux", "chroma"]).required(),
      flux_only: Schema.boolean(),
      apply_t5_attn_mask: Schema.boolean().default(true),
    }),
    Schema.object({
      model_type: Schema.const("anima").required(),
      anima_only: Schema.boolean(),
    }),
    Schema.object({}),
  ]),
  Schema.object({
    model_train_type: Schema.string().default("flux-lora"),
    optimizer_type: Schema.union(["AdamW", "AdamW8bit"]).default("AdamW8bit"),
    lr_scheduler: Schema.union(["constant", "cosine"]).default("constant"),
  }),
])
'''


SD_SYNTHETIC = r'''
Schema.intersect([
  Schema.object({
    model_train_type: Schema.union(["sd-dreambooth", "sdxl-finetune"]).default("sd-dreambooth"),
    pretrained_model_name_or_path: Schema.string(),
  }),
  Schema.union([
    Schema.object({
      model_train_type: Schema.const("sd-dreambooth").required(),
      v2: Schema.boolean(),
      learning_rate_te: Schema.number(),
    }),
    Schema.object({
      model_train_type: Schema.const("sdxl-finetune").required(),
      learning_rate_te1: Schema.number(),
      learning_rate_te2: Schema.number(),
    }),
    Schema.object({}),
  ]),
  Schema.object({
    optimizer_type: Schema.union(["AdamW", "AdamW8bit"]).default("AdamW8bit"),
    lr_scheduler: Schema.union(["constant", "cosine"]).default("constant"),
  }),
])
'''


def execute_schema(source: str) -> dict:
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node is required for Schemastery runtime contract")
    script = SCHEMA_STUB + "\nconst result = " + source + ";\n" + r'''
const keys = collectKeys(result);
process.stdout.write(JSON.stringify({keys}));
'''
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class TrainingSchemaFactoryRuntimeTests(unittest.TestCase):
    def assert_no_routing_fields(self, keys):
        self.assertNotIn("model_type", keys)
        self.assertNotIn("model_train_type", keys)
        self.assertNotIn("anima_training_mode", keys)

    def test_anima_finetune_prunes_flux_lora_and_all_routing_fields(self):
        source = fixed_flux_family_schema(
            FLUX_FAMILY_SYNTHETIC,
            model_type="anima",
            train_type="anima-finetune",
            anima_mode="finetune",
        )
        keys = execute_schema(source)["keys"]

        self.assertIn("anima_model_variant", keys)
        self.assertIn("qwen3", keys)
        self.assertIn("vae", keys)
        self.assertIn("self_attn_lr", keys)
        self.assertIn("train_qwen3_text_encoder", keys)
        self.assertIn("anima_only", keys)
        self.assertIn("optimizer_type", keys)
        self.assertIn("lr_scheduler", keys)
        self.assertNotIn("ae", keys)
        self.assertNotIn("clip_l", keys)
        self.assertNotIn("t5xxl", keys)
        self.assertNotIn("network_dim", keys)
        self.assertNotIn("flux_only", keys)
        self.assert_no_routing_fields(keys)

    def test_anima_lora_prunes_full_branch_and_routing_fields(self):
        source = fixed_flux_family_schema(
            FLUX_FAMILY_SYNTHETIC,
            model_type="anima",
            train_type="anima-lora",
            anima_mode="lora",
        )
        keys = execute_schema(source)["keys"]
        self.assertIn("anima_model_variant", keys)
        self.assertIn("network_dim", keys)
        self.assertIn("optimizer_type", keys)
        self.assertNotIn("self_attn_lr", keys)
        self.assertNotIn("train_qwen3_text_encoder", keys)
        self.assertNotIn("flux_only", keys)
        self.assert_no_routing_fields(keys)

    def test_flux_page_prunes_anima_fields_and_routing_fields(self):
        source = fixed_flux_family_schema(
            FLUX_FAMILY_SYNTHETIC,
            model_type="flux",
            train_type="flux-lora",
        )
        keys = execute_schema(source)["keys"]
        self.assertIn("ae", keys)
        self.assertIn("clip_l", keys)
        self.assertIn("t5xxl", keys)
        self.assertIn("flux_only", keys)
        self.assertIn("optimizer_type", keys)
        self.assertNotIn("anima_model_variant", keys)
        self.assertNotIn("qwen3", keys)
        self.assertNotIn("anima_only", keys)
        self.assert_no_routing_fields(keys)

    def test_chroma_page_is_t5_only_without_hidden_routing_fields(self):
        source = fixed_flux_family_schema(
            FLUX_FAMILY_SYNTHETIC,
            model_type="chroma",
            train_type="chroma-lora",
        )
        keys = execute_schema(source)["keys"]
        self.assertIn("ae", keys)
        self.assertIn("t5xxl", keys)
        self.assertIn("flux_only", keys)
        self.assertIn("optimizer_type", keys)
        self.assertNotIn("clip_l", keys)
        self.assertNotIn("apply_t5_attn_mask", keys)
        self.assertNotIn("anima_model_variant", keys)
        self.assert_no_routing_fields(keys)

    def test_sd_dreambooth_does_not_inherit_sdxl_or_routing_field(self):
        source = fixed_sd_schema(SD_SYNTHETIC, "sd-dreambooth")
        keys = execute_schema(source)["keys"]
        self.assertIn("v2", keys)
        self.assertIn("learning_rate_te", keys)
        self.assertIn("optimizer_type", keys)
        self.assertIn("lr_scheduler", keys)
        self.assertNotIn("learning_rate_te1", keys)
        self.assertNotIn("learning_rate_te2", keys)
        self.assert_no_routing_fields(keys)


if __name__ == "__main__":
    unittest.main()
