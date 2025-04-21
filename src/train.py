import tempfile
import time

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss_fn):
    if torch.cuda.is_available():
        model = model.cuda()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss_value = loss_fn(output, target)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def valid_one_epoch(valid_dataloader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss_value = loss_fn(output, target)
            running_loss += loss_value.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = running_loss / len(valid_dataloader)
        accuracy = 100. * correct / total

    return avg_loss, accuracy


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    for epoch in range(1, n_epochs + 1):
        print(f"\n🔁 Epoch {epoch}/{n_epochs}")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        val_loss, val_acc = valid_one_epoch(data_loaders["valid"], model, loss)

        epoch_time = time.time() - start_time
        print(
            f"⏱️ Duration: {epoch_time:.2f}s | "
            f"📈 Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% | "
            f"🧪 Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%"
        )

        # Save model if validation improves
        if valid_loss_min is None or ((valid_loss_min - val_loss) / valid_loss_min > 0.01):
            print(f"💾 New best model with val loss {val_loss:.6f}. Saving to {save_path}")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = val_loss

        scheduler.step(val_loss)

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = val_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    test_loss = 0.
    correct = 0.
    total = 0.

    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            logits = model(data)
            loss_value = loss(logits, target)
            test_loss += loss_value.item()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += torch.sum(pred.eq(target.view_as(pred))).item()
            total += target.size(0)

    avg_test_loss = test_loss / len(test_dataloader)
    accuracy = 100. * correct / total

    print(f'\n📊 Test Loss: {avg_test_loss:.6f}')
    print(f'✅ Test Accuracy: {accuracy:.2f}% ({int(correct)}/{int(total)})')

    return avg_test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt, acc = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lv, acc = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
