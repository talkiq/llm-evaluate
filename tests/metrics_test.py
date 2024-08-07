from voiceai.nlp.llm_evaluate.datasets import EvaluationDataset
from voiceai.nlp.llm_evaluate.datasets import load_dataset
from voiceai.nlp.llm_evaluate.helpers.profile import Profile
from voiceai.nlp.llm_evaluate.models import Model


def test_metrics(simple_model: Model, default_profile: Profile) -> None:
    batch_size = 1
    catalog = default_profile.datasets
    assert catalog

    # check for datasets
    for name in catalog:
        dataset = load_dataset(name, default_profile, batch_size=batch_size)
        assert isinstance(dataset, EvaluationDataset)
        dataloader = dataset.get_dataloader()
        for batch in dataloader:
            inputs, _ = batch
            assert len(inputs) == batch_size
            assert isinstance(inputs, (list, tuple))
            assert isinstance(inputs[0], str)
            outputs = simple_model.process(inputs)
            assert isinstance(outputs, list)
            assert isinstance(outputs[0], str)
