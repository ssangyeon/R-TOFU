import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    # forget_loss
    if 'GA1' in loss_type:
        forget_loss = ga1_loss(model, inputs)
    elif 'GA2' in loss_type:
        forget_loss = ga2_loss(model, inputs)
    elif 'GA3' in loss_type:
        forget_loss = ga3_loss(model, inputs)
    elif 'FINETUNE' in loss_type:
        forget_loss = finetune_loss(model,inputs)
    elif 'IDK1' in loss_type:
        forget_loss = idk1_loss(model, inputs)
    elif 'IDK2' in loss_type:
        forget_loss = idk2_loss(model, inputs)
    elif 'IDK3' in loss_type:
        forget_loss = idk3_loss(model, inputs)

    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    elif 'KL' in loss_type:
        regularization_loss = kl_loss(model, ref_model, inputs)
    else:
        regularization_loss = 0

    return forget_loss, regularization_loss


def finetune_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = +1 * outputs.loss
    return loss

# Forget Loss
def ga1_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga2_loss(model, inputs):
    forget_inputs = inputs[1]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga3_loss(model, inputs):
    forget_inputs = inputs[2]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss



def idk1_loss(model, inputs):
    forget_idk_inputs = inputs[3]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def idk2_loss(model, inputs):
    forget_idk_inputs = inputs[4]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def idk3_loss(model, inputs):
    forget_idk_inputs = inputs[5]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss



# Regularization Loss
def gd_loss(model, inputs):
    retain_inputs = inputs[6]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[6]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits.detach()
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss


