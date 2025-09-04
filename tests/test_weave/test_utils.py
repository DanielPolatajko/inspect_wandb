from inspect_wandb.weave.utils import format_score_types, format_sample_display_name
from inspect_wandb.shared.utils import format_wandb_id_string as format_model_name
import pytest
import re

@pytest.mark.parametrize("model_name", [
    "google/vertex/gemini-2.0-flash",
    "anthropic/vertex/claude-3-5-sonnet-v2@20241022",
    "anthropic/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic/claude-3-5-sonnet-latest",
    "hf/openai-community/gpt2"
])
def test_format_model_name_correctly_formats_inspect_valid_model_names(model_name: str) -> None:
    formatted_model_name = format_model_name(model_name)
    assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", formatted_model_name)


class TestFormatScoreTypes:
    """Test cases for format_score_types function."""
    
    def test_string_input(self):
        """Test that string input is wrapped in a score dict."""
        result = format_score_types("0.85")
        assert result == {"score": "0.85"}
    
    def test_int_input(self):
        """Test that int input is converted to float."""
        result = format_score_types(85)
        assert result == 85.0
        assert isinstance(result, float)
    
    def test_float_input(self):
        """Test that float input is returned as-is."""
        result = format_score_types(0.85)
        assert result == 0.85
    
    def test_single_element_sequence(self):
        """Test that single-element sequence is wrapped in a score dict."""
        result = format_score_types([0.85])
        assert result == {"score": 0.85}
    
    def test_single_element_tuple(self):
        """Test that single-element tuple is wrapped in a score dict."""
        result = format_score_types((0.85,))
        assert result == {"score": 0.85}
    
    def test_multi_element_sequence_raises_error(self):
        """Test that multi-element sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence score cannot be passed to Weave"):
            format_score_types([0.85, 0.90])
    
    def test_empty_sequence_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence score cannot be passed to Weave"):
            format_score_types([])
    
    def test_mapping_input(self):
        """Test that mapping input is converted to dict."""
        input_dict = {"accuracy": 0.85, "precision": 0.90}
        result = format_score_types(input_dict)
        assert result == input_dict
        assert isinstance(result, dict)
    
    def test_dict_with_score_key(self):
        """Test that dict with score key is returned as-is."""
        input_dict = {"score": 0.85}
        result = format_score_types(input_dict)
        assert result == input_dict
    
    def test_other_types_returned_as_is(self):
        """Test that other types (like bool) are converted to appropriate types."""
        assert format_score_types(True) == 1.0
        assert format_score_types(False) == 0.0
    
    def test_complex_nested_mapping(self):
        """Test that complex nested mappings are handled correctly."""
        input_dict = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.90
            },
            "score": 0.87
        }
        result = format_score_types(input_dict)
        assert result == input_dict


class TestFormatSampleDisplayName:
    """Test cases for format_sample_display_name function."""
    
    def test_basic_template_substitution(self):
        """Test basic template variable substitution."""
        template = "{task_name}-sample-{sample_id}-epoch-{epoch}"
        result = format_sample_display_name(template, "test_task", 1, 1)
        assert result == "test_task-sample-1-epoch-1"
    
    def test_custom_template_format(self):
        """Test custom template formats work correctly."""
        template = "{task_name}_s{sample_id}_e{epoch}"
        result = format_sample_display_name(template, "my_task", 42, 3)
        assert result == "my_task_s42_e3"
    
    def test_minimal_template(self):
        """Test minimal template with only one variable."""
        template = "Sample {sample_id}"
        result = format_sample_display_name(template, "task", 123, 5)
        assert result == "Sample 123"
    
    def test_complex_template(self):
        """Test complex template with text and multiple variables."""
        template = "Task: {task_name} | ID: {sample_id} | Epoch: {epoch}"
        result = format_sample_display_name(template, "classification", 99, 10)
        assert result == "Task: classification | ID: 99 | Epoch: 10"
    
    def test_invalid_template_fallback(self):
        """Test that invalid templates fall back to default format."""
        template = "{task_name}-{invalid_var}-{sample_id}"
        result = format_sample_display_name(template, "test", 1, 1)
        assert result == "test-sample-1-epoch-1"
    
    def test_empty_template_fallback(self):
        """Test that empty template falls back to default format."""
        template = ""
        result = format_sample_display_name(template, "test", 1, 1)
        assert result == "test-sample-1-epoch-1"
    
    def test_malformed_template_fallback(self):
        """Test that malformed templates fall back gracefully."""
        template = "{task_name}-{unclosed_brace"
        result = format_sample_display_name(template, "test", 1, 1)
        assert result == "test-sample-1-epoch-1"
    
    def test_special_characters_in_values(self):
        """Test that special characters in values are handled correctly."""
        template = "{task_name}-{sample_id}"
        result = format_sample_display_name(template, "test-with-dashes", 1, 1)
        assert result == "test-with-dashes-1"
    
    def test_numeric_values_formatting(self):
        """Test that numeric values are formatted correctly."""
        template = "{task_name}-{sample_id}-{epoch}"
        result = format_sample_display_name(template, "task", 0, 999)
        assert result == "task-0-999"
    
    def test_string_sample_id(self):
        """Test that string sample IDs work correctly."""
        template = "{task_name}-{sample_id}-{epoch}"
        result = format_sample_display_name(template, "task", "sample_123", 1)
        assert result == "task-sample_123-1"
    
    @pytest.mark.parametrize("template,task_name,sample_id,epoch,expected", [
        ("{task_name}", "simple", 1, 1, "simple"),
        ("{sample_id}", "task", 42, 1, "42"),
        ("{sample_id}", "task", "str_id", 1, "str_id"),
        ("{epoch}", "task", 1, 7, "7"),
        ("{task_name}_{sample_id}_{epoch}", "eval", 10, 2, "eval_10_2"),
        ("{task_name}_{sample_id}_{epoch}", "eval", "abc123", 2, "eval_abc123_2"),
        ("prefix-{task_name}-suffix", "test", 1, 1, "prefix-test-suffix"),
    ])
    def test_template_variations(self, template, task_name, sample_id, epoch, expected):
        """Test various template patterns."""
        result = format_sample_display_name(template, task_name, sample_id, epoch)
        assert result == expected