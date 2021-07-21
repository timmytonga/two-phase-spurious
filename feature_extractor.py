from tqdm.auto import tqdm
import torch


class FeatureExtractor:
    """
    Given a network, a layer's name, and a dataset, return or save the corresponding feature vectors
    along with the model's prediction on the given dataset.
    """

    def __init__(self, model, device, layers_name='fc', get_layers_input=False):
        """
        model: is a preloaded PyTorch model that we will extract the features from the layers_name layer
            using the extract features function below.
        device: whether things are getting processed on gpus or cpus.
        layers_name: is the layer's string where it has to be a valid attribute of the model
            that we want to extract the feature from. Default to the final layer.
        get_layers_input: Boolean indicating whether the feature of the layer is its input or output.
        """
        self.model = model
        self.layers_name = layers_name
        self.get_layers_input = get_layers_input
        self.device = device

    def extract_features(self, dataloader, show_progress=True):
        """
            Return output sets (see structure below) given a data_loader using the model
        """
        output_sets = {'activations': [],  # the features
                       'predicted': [],  # the model's prediction
                       'labels': [],  # the corresponding labels
                       'idx': []  # the idx of the corresponding activations (to match with original data)
                       }
        self.model.eval()  # do not collect stats
        # the line below makes it so that whenever model.forward is called, we save the features to our activations
        # list. the feature handler ensures we remove the hook after we are done extracting the features
        feature_handler = self._register_forward_hook_for_layer(self.layers_name, output_sets['activations'])
        loader_bar = tqdm(dataloader) if show_progress else dataloader

        with torch.set_grad_enabled(False):  # important so we don't waste a bunch of GPU memories
            for batch_idx, batch in enumerate(loader_bar):
                batch = tuple(t.to(self.device) for t in batch)
                x, y, g, idx = batch
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)

                output_sets['labels'].append(y.detach().cpu())
                output_sets['predicted'].append(predicted.detach().cpu())
                output_sets['idx'].append(idx.detach().cpu())

        feature_handler.remove()  # we unregister the forward hook.
        return output_sets

    def _register_forward_hook_for_layer(self, layer_name, save_list):
        """
        Register a forward hook for the given layer's name: whenever we call model.forward, we save the input/output of
            the layer to the save_list. IMPORTANT: must unregister the forward_hook later to avoid weird behaviors.
            More here: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
        """
        assert layer_name is not None
        if self.get_layers_input:
            def hook_fn(_, inp, _outp):
                save_list.append(inp[0].detach().cpu())  # this saves the feature to the given feature list
        else:
            def hook_fn(_, _inp, outp):
                save_list.append(outp.detach().cpu())

        for name, m in self.model.named_modules():
            if name == layer_name:
                return m.register_forward_hook(hook_fn)
        raise NameError(f"Cannot find {layer_name} in given model (maybe just because DataParallel is not implemented)")
