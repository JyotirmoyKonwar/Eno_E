# DarkGuard: Training Language Models to Detect Dark Patterns

*Our OpenEnv environment and two-stage post-training pipeline for turning a general model into a stronger dark-pattern detector.*

LLMs are usually great for generation tasks. But far fewer can reliably detect dark patterns in a structured, repeatable way inside a real training loop that too in a zero/few shot manner. That gap is what led us to build **DarkGuard**. Dark patterns on internet are super common, encountered by almost everyone, and slightly annoying for everyone!

DarkGuard is an OpenEnv environment for training language models to identify dark patterns: user interface and product design choices that pressure, confuse, misdirect, or manipulate people into actions they may not have intended to take. Instead of treating this as a one-shot classification task, we frame it as an environment where the model must analyze a scenario, respond in a valid format, and improve through reward-driven training.

We narrowed down our goal . We were not trying to build a general safety system, broad cybersecurity product, or a vague alignment benchmark. We focused on one problem only: **dark pattern detection**. From one of our previous experience in the Dark Patterns Buster Hackathon Organised in 2023.

## Why dark pattern detection needs an environment

Dark patterns are harder than they first appear. A model is not simply looking for a suspicious keyword or a single bad sentence unlike normal text classification models. It has to reason about how a design is steering a user. 

That means looking at things like:
- whether one option is visually or linguistically different frm the other,
- whether urgency is being used to pressurize the user,
- whether decline paths are hidden, harder, or more confusing,
- and whether the overall interaction is fair or manipulative.

A simple classifier can sometimes catch obvious cases, but it often struggles with borderline or contextual ones and can't reason across steps. We wanted a setup where the model is not just asked, “Is this bad?” We wanted a setup where it has to operate inside a structured task, produce valid outputs, and be trainable through reinforcement learning.

## What is DarkGuard

![DarkGuard Environment Diagram](environment_structure.png)



DarkGuard is built as an OpenEnv environment for dark pattern detection. The environment presents the model with structured scenarios and evaluates the response through a rewardable interaction loop.

The model is expected to do more than just generate fluent text. It must:
- interpret the scenario correctly,
- detect whether a dark pattern is present,
- respond in the expected format,
- and stay within the environment’s validity constraints.

This distinction matters. A model can sound intelligent and still fail the task. In DarkGuard, outputs must be both useful and valid inside the environment. That makes the setup much better for training than a loose prompt-and-judge workflow.

We also wanted the environment to be practical for actual RL. That meant keeping the problem specific, making outputs checkable, and ensuring the model had a real path to improvement rather than being thrown into a task that was too vague or too hard and optimizing for reward function exploitation.

## For our training POS we resorted to a two-stage fine-tuning pipeline 

To train the model effectively, we used a **two-stage fine-tuning process** over Qwen3-4B-Thinking-2507-FP8 for relatively faster training.

### Stage 1: SFT for a unified model adapter

The first stage was supervised fine-tuning to create a **unified model adapter** for dark pattern reasoning.

Although datasets for our specific RL focused use case didn't exist, we decided that model needed to understand Dark Patterns well before detection as well as generation of these patterns, which was part of our multi-agent self-play plan! 

The target of this stage was to learn how to read the scenarios, how to structure its outputs, and how to respond in a way that the environment could reliably interpret. In other words, SFT gave us a model that could consistently “show up” to the task in the right way. Essentially our RL loop environment relied on the model's ability to understand the dark patterns, come up with different ones themselves and detect them.

However during the SFT it was obvious, that loss function of Generator was very less compared to the mediocre performance of the Consumer (detector model). The idea being even after the multiple checks, generator can still produce less relevant dark patterns and pass them as the pattern while for the Detectors, they have to be pretty accurate with predictions. Adding to that, not enough time to refine the different datasets to the specific usecase we are aiming for. 

SFT was still important as Reinforcement learning works much better when the model already has some baseline competence. If the model cannot produce valid or partially correct outputs, then reward becomes too sparse and training becomes unstable.

So the SFT phase was the foundation.

### Stage 2: GRPO on the DarkGuard environment

Once we had as stable as we could unified adapter, we moved to the second stage: **GRPO training directly on the DarkGuard environment**. We earlier experimented with different weights for Generator and Consumer models, essentially to check how performance varies across such small models, and would this improve or decrease performance for our models. It was noticed that the generator model was miles above performance of our consumer model hence, we decided to prompt seperate these two models from a single checkpoint, so as both model have equal understanding of what dark pattern is, and the prompt decdes what to do with the understanding and reasoning.

This is where the model should stop imitating examples and start improving its policy based on outcomes inside the environment. 

That is the aim of the Darkguard OpenEnv.

The GRPO stage let the model:
- improve beyond surface formatting,
- make stronger decisions inside the environment,
- become more consistent under the task-specific reward structure,
- and learn behavior that was actually rewarded by the environment rather than just looking good in isolation.
- and ultimately adapt to consistent improvement in dark patterns, understanding the gist of it step by step. 

In short, the first stage taught the model how to participate in the task. The second stage taught it how to get better at the task.

![DarkGuard Simple Diagram](simple_training_idea.png)
---
![DarkGuard Detail Training Diagram](training_loop.png)


## What the training results show

The most useful part of this project is that we can point to real training behavior, not just a claimed pipeline.

Across the run, several trends stood out.


### 1. ELO improved on both sides

The clearest performance signal in the dashboard is the upward movement in ELO, while Consumer model clearly struggled initially and only reached mere 30-50 points above baseline in the first 500 steps, the Generator model consistently beat the Consumer in self-play essentially, meaning consumer task needs improvement and better training or Generator should be more constricted and accurate in newer generations of metadata.

The **consumer ELO** shows early instability, followed by recovery and then a steady climb across training. The **designer ELO** also rises substantially and more smoothly over time. Taken together, these curves indicate that the trained model became stronger relative to its comparison pool as training progressed.

This is important because ELO is easier to interpret than raw RL loss alone. Loss can be noisy. Relative performance is often a clearer story. ELO is relative performance to initial model and in that sense, the models have definitely improved. 


### 2. Invalid rates dropped sharply after the early phase

One of the most encouraging patterns in the current run is the drop in invalid behavior. Earlier our environment struggled with invalid outputs due to which we moved to SFT and more refining and constraints in environment so both models don't cheat on their turns.

Even in this run, both training-time and evaluation-time invalid rates are noisy at the start, with several large spikes early in the run. But after that initial phase, they drop close to zero for most of training.

That is one of the thing RL teaches the model, with more episodes, it becomes much better at staying within the rules of the environment.


### 3. Reward-side metrics were strict, not inflated

Some of the reward curves remain noisy and conservative. Mean reward and evaluation reward stay negative for much of the run, even while ELO and validity improve.

We see this as a sign that the environment is not handing out easy wins.

A strict reward setup is often better than an inflated one to actually understand improvement, essential to our self-play focus on this environment. It means the model is being pushed by a hard objective rather than coasting on a permissive score. In our case, the combination of improving ELO, dropping invalidity, and conservative reward values suggests that the model is learning under real pressure rather than benefiting from a soft metric.

### 4. False-positive behavior stayed relatively controlled

The false-positive rate remains low for most of training, with only occasional spikes.

That is an important result for dark pattern detection. A detector that flags everything as manipulative is not useful. So the rewards were structured so as to handle these different cases of Reward function exploitations i.e. Reward hacking.

Keeping false positives relatively controlled suggests that the model is learning something more balanced than simply calling every scenario a dark pattern.


Ultimately, even if the model was small and the environment task was pretty hard for a 4B model to learn (to learn detection as well as accurate generation!) 

In our run, we see:
- ELO rises,
- invalid rates fall,
- validity stays stable,
- and reward-facing behavior remains meaningful.


## What we learned

This project reinforced a few practical lessons.

First, **dark pattern detection can benefit from structured training**, not just prompting. Once we turned the task into an environment, it became much easier to reason about model behavior, reward design, and measurable improvement for more complex patterns which simple chat models can't reason through.

Second, **SFT and RL played different roles**. SFT gave us a unified adapter and stable behavior. GRPO gave us policy improvement inside the environment. Trying to skip the first step would likely have made the second one much less stable. While SFT definitely helps more, in understanding the pattern, RL environment is a crucial part for use cases where simple imitation based detection won't work and adaptability around newer examples and creativity and understanding to create/understand is there.


## What DarkGuard shows

DarkGuard shows that dark pattern detection can be trained as an actual environment task rather than treated only as static labeling.

The project demonstrates a full pipeline:
- an OpenEnv-based environment,
- a supervised warm-start through a unified adapter,
- GRPO training on the real task,
- and measurable changes in competitive and behavioral metrics.
- Learn through self-play and create own experiences.

## Closing

DarkGuard is a focused project with a focused claim.

It is an environment for training language models to detect dark patterns. We used a two-stage pipeline, SFT for a unified model adapter, followed by GRPO on the DarkGuard environment, to move from task formatting and baseline competence to measurable policy improvement.

While results are simple, they're also pretty encouraging since a simple model is able to learn and do such complex tasks. The trained model becomes more valid, more competitive, and more consistent inside the environment. ELO improves, invalid behavior drops after the early phase, and the system shows real evidence of learning rather than just better wording.

This is our main contribution to the OpenEnv Environment through this hackathon project.

