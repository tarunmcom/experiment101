# Slack Notifications Guide

The training script now sends automatic Slack notifications to keep you updated on training progress!

## 📱 Notification Types

### 1. Training Start (Sent once at beginning)
```
🚀 *U-Net Training Started*
• Device: cuda
• Training samples: 1200
• Validation samples: 300
• Batch size: 8
• Max epochs: 100
• Learning rate: 0.001
• Model parameters: 31,042,945
```

### 2. Periodic Updates (Every N epochs)
```
📊 *Epoch 10/100 Update*
• Train Loss: 0.1234 | Val Loss: 0.1567
• Train IoU: 0.8567 | Val IoU: 0.8234
• Train Dice: 0.8901 | Val Dice: 0.8567
• Learning Rate: 1.00e-03
```

### 3. Best Model Saved (When IoU improves)
```
🎯 *New Best Model! (IoU)*
• Epoch: 25/100
• Val IoU: 0.8456
• Val Loss: 0.1234
• Val Dice: 0.8789
```

### 4. Early Stopping (If triggered)
```
⛔ *Early Stopping Triggered*
• Stopped at epoch: 65/100
• Best Val IoU: 0.8456
• Best Val Loss: 0.1234
• No improvement for 15 epochs
```

### 5. Training Complete (Final summary)
```
✅ *Training Completed!*
• Total epochs: 65
• Best Val IoU: 0.8456
• Best Val Loss: 0.1234
• Final Train IoU: 0.8901
• Final Val IoU: 0.8456
• Model saved in: checkpoints/
🎉 Ready for inference!
```

## ⚙️ Configuration

All Slack settings are in **lines 501-504** of `train_unet.py`:

```python
# Slack notifications
'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
'slack_enabled': True,  # Set to False to disable notifications
'slack_update_frequency': 5,  # Send update every N epochs
```

## 🔧 How to Configure

### 1. Update Webhook URL (Already Set!)

Your webhook URL is already configured:
```python
'slack_webhook': 'https://hooks.slack.com/services/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'
```

### 2. Enable/Disable Notifications

```python
# Enable notifications
'slack_enabled': True,

# Disable notifications (no messages sent)
'slack_enabled': False,
```

### 3. Change Update Frequency

```python
# Send updates every 5 epochs (default)
'slack_update_frequency': 5,

# Send updates every epoch (more frequent)
'slack_update_frequency': 1,

# Send updates every 10 epochs (less frequent)
'slack_update_frequency': 10,
```

## 📊 What You'll Receive

### Typical Training Session (100 epochs)

| Epoch | Notification Type | Message |
|-------|------------------|---------|
| 0 | Start | 🚀 Training Started |
| 5 | Periodic Update | 📊 Epoch 5/100 Update |
| 8 | Best Model | 🎯 New Best Model! (IoU) |
| 10 | Periodic Update | 📊 Epoch 10/100 Update |
| 15 | Periodic Update | 📊 Epoch 15/100 Update |
| 18 | Best Model | 🎯 New Best Model! (IoU) |
| 20 | Periodic Update | 📊 Epoch 20/100 Update |
| ... | ... | ... |
| 65 | Early Stop | ⛔ Early Stopping Triggered |
| 65 | Complete | ✅ Training Completed! |

**Expected notifications for a full run:**
- 1× Training Start
- ~13-20× Periodic Updates (depending on total epochs)
- ~3-8× Best Model notifications (as model improves)
- 1× Early Stopping OR Training Complete

**Total: ~20-30 messages** over the entire training session (1-2 hours)

## 🎯 Smart Features

### 1. Error Handling
If Slack is unavailable, training continues without interruption:
```python
try:
    send_slack(msg, webhook_url)
except Exception as e:
    print(f"⚠️  Failed to send Slack notification: {e}")
    # Training continues normally!
```

### 2. Timeout Protection
Slack requests timeout after 5 seconds to prevent blocking training.

### 3. Optional Feature
Notifications are completely optional - disable anytime without affecting training.

## 📱 Notification Schedule

### Example with `slack_update_frequency: 5`

```
[Start]      🚀 Training Started
[Epoch 5]    📊 Periodic Update
[Epoch 10]   📊 Periodic Update + 🎯 Best Model (if improved)
[Epoch 15]   📊 Periodic Update
[Epoch 20]   📊 Periodic Update
[Epoch 25]   📊 Periodic Update + 🎯 Best Model (if improved)
...
[End]        ✅ Training Completed!
```

### Example with `slack_update_frequency: 1` (Every epoch)

```
[Start]      🚀 Training Started
[Epoch 1]    📊 Periodic Update
[Epoch 2]    📊 Periodic Update + 🎯 Best Model (if improved)
[Epoch 3]    📊 Periodic Update
[Epoch 4]    📊 Periodic Update + 🎯 Best Model (if improved)
...
[End]        ✅ Training Completed!
```

## 💡 Pro Tips

### 1. For Long Training (100+ epochs)
```python
'slack_update_frequency': 10,  # Less frequent, less spam
```

### 2. For Quick Experiments (10-20 epochs)
```python
'slack_update_frequency': 5,  # Default, good balance
```

### 3. For Close Monitoring
```python
'slack_update_frequency': 1,  # Every epoch
```

### 4. For Debugging (No notifications)
```python
'slack_enabled': False,  # Disable completely
```

## 🔍 What Metrics Are Tracked?

All notifications include key metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Train Loss** | Training set loss | Lower is better (< 0.15) |
| **Val Loss** | Validation set loss | Lower is better (< 0.20) |
| **Train IoU** | Training Intersection over Union | Higher is better (> 0.80) |
| **Val IoU** | Validation Intersection over Union | Higher is better (> 0.75) |
| **Train Dice** | Training Dice Coefficient | Higher is better (> 0.85) |
| **Val Dice** | Validation Dice Coefficient | Higher is better (> 0.80) |
| **Learning Rate** | Current learning rate | Decreases over time |

## 🎨 Emoji Guide

| Emoji | Meaning |
|-------|---------|
| 🚀 | Training started |
| 📊 | Periodic update |
| 🎯 | New best model saved |
| ⛔ | Early stopping triggered |
| ✅ | Training completed successfully |
| 🎉 | Ready for next step |

## 🔧 Customization

Want to customize the messages? Edit the `send_slack()` calls in `train_unet.py`:

### Example: Add more details to periodic updates

Find line ~638 and modify:
```python
msg = f"📊 *Epoch {epoch}/{config['num_epochs']} Update*\n" \
      f"• Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}\n" \
      f"• Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}\n" \
      f"• Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}\n" \
      f"• Learning Rate: {current_lr:.2e}\n" \
      f"• Best so far: {best_val_iou:.4f}"  # ← Add this line
```

### Example: Add custom message

```python
msg = f"🎯 New Best Model!\n" \
      f"🏆 This is better than before!\n" \
      f"📈 Keep going!"
```

## 🚨 Troubleshooting

### Notifications not working?

1. **Check webhook URL is correct**
   - Line 502 in `train_unet.py`
   - Should start with `https://hooks.slack.com/services/`

2. **Check notifications are enabled**
   ```python
   'slack_enabled': True,  # Must be True
   ```

3. **Check internet connection**
   - Slack requires internet access
   - Training continues even if Slack fails

4. **Test webhook manually**
   ```python
   python senslackdata.py  # Test your webhook
   ```

### Too many notifications?

```python
# Increase frequency (fewer messages)
'slack_update_frequency': 10,  # Instead of 5

# Or disable periodic updates, keep only important ones
# (requires code modification)
```

### Not enough notifications?

```python
# Decrease frequency (more messages)
'slack_update_frequency': 1,  # Every epoch
```

## 📋 Quick Reference

| Configuration | Line | Default | Options |
|--------------|------|---------|---------|
| Webhook URL | 502 | Your URL | Any Slack webhook |
| Enable/Disable | 503 | `True` | `True` or `False` |
| Update Frequency | 504 | `5` | Any integer (1, 5, 10, etc.) |

## 🎯 Summary

**What you get:**
- ✅ Real-time training updates on Slack
- ✅ No need to watch terminal constantly
- ✅ Get notified on phone/desktop
- ✅ Track progress remotely
- ✅ Know immediately when training completes

**How to use:**
1. Webhook URL is already configured ✓
2. Notifications are enabled by default ✓
3. Just run `python train_unet.py` ✓
4. Check your Slack for updates! ✓

**That's it!** Your training will now keep you updated via Slack! 🎉

---

**Note:** All notifications are non-blocking. If Slack fails, training continues normally without interruption.

