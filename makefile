PYTHON := python
WEIGHT_DIR := model_weights
PUSH_SCRIPT := scripts/push_model_weights.py
PULL_SCRIPT := scripts/pull_model_weights.py
MAIN_SCRIPT := main.py

MODEL_ARGS :=

# define experiment prefixes for model1-model6
MODEL1_PREFIX := experiment_1
MODEL2_PREFIX := experiment_2
MODEL3_PREFIX := experiment_3
MODEL4_PREFIX := experiment_4
MODEL5_PREFIX := experiment_5
MODEL6_PREFIX := experiment_6

define ensure_weights_exist
	mkdir -p $(WEIGHT_DIR); \
	MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/$(1)*.pt 2>/dev/null | head -n1); \
	if [ -z "$$MODEL_PATH" ]; then \
		echo "Pulling $(1) from Hugging Face..."; \
		$(PYTHON) $(PULL_SCRIPT) --prefix $(1) --output_dir $(WEIGHT_DIR); \
	fi; \
	echo "Waiting for $(1) weights to be ready to load..."; \
	while true; do \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/$(1)*.pt 2>/dev/null | head -n1); \
		if [ -f "$$MODEL_PATH" ]; then break; fi; \
		sleep 1; \
	done; \
	echo "weights ready: $$MODEL_PATH"
endef

# these commands will choose the latest weights for each model
model_1:
	@bash -c '\
		$(call ensure_weights_exist,experiment_1); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_1*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name gpt2 \
			--image_encoder_type vit \
			--decoder_type gpt2 \
			--d_model 768 \
			--train_type finetune \
			--llava_projections; \
	'

model_2:
	@bash -c '\
		$(call ensure_weights_exist,experiment_2); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_2*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name lmsys/vicuna-7b-v1.5 \
			--image_encoder_type vit \
			--decoder_type vicuna \
			--d_model 4096 \
			--train_type lora \
			--llava_projections; \
	'

model_3:
	@bash -c '\
		$(call ensure_weights_exist,experiment_3); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_3*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name gpt2 \
			--image_encoder_type clip \
			--decoder_type gpt2 \
			--d_model 768 \
			--train_type finetune \
			--llava_projections; \
	'

model_4:
	@bash -c '\
		$(call ensure_weights_exist,experiment_4); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_4*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name lmsys/vicuna-7b-v1.5 \
			--image_encoder_type clip \
			--decoder_type vicuna \
			--d_model 4096 \
			--train_type lora \
			--llava_projections; \
	'

model_5:
	@bash -c '\
		$(call ensure_weights_exist,experiment_5); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_5*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name lmsys/vicuna-7b-v1.5 \
			--image_encoder_type clip \
			--decoder_type vicuna \
			--d_model 4096 \
			--train_type mlp-pretrain \
			--llava_projections; \
	'

model_6:
	@bash -c '\
		$(call ensure_weights_exist,experiment_6); \
		MODEL_PATH=$$(ls -t $(WEIGHT_DIR)/experiment_6*.pt 2>/dev/null | head -n1); \
		$(PYTHON) $(MAIN_SCRIPT) \
			--model_path $$MODEL_PATH \
			--tokenizer_name lmsys/vicuna-7b-v1.5 \
			--image_encoder_type vit \
			--decoder_type vicuna \
			--d_model 4096 \
			--train_type mlp-pretrain \
			--llava_projections; \
	'

push_weights:
	@echo "Pushing all model weights..."
	@$(PYTHON) $(PUSH_SCRIPT)

all_weights:
	@echo "Pulling latest weights for all models..."
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL1_PREFIX) --output_dir $(WEIGHT_DIR)
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL2_PREFIX) --output_dir $(WEIGHT_DIR)
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL3_PREFIX) --output_dir $(WEIGHT_DIR)
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL4_PREFIX) --output_dir $(WEIGHT_DIR)
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL5_PREFIX) --output_dir $(WEIGHT_DIR)
	@$(PYTHON) $(PULL_SCRIPT) --prefix $(MODEL6_PREFIX) --output_dir $(WEIGHT_DIR)
