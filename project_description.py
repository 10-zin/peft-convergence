"""
Accelerating PEFT Convergence in Low-Data Regimes via Optimizer & Schedule Tuning
Tenzin Bhotia, Tanvi Kaple, Bhargava Somu, Prateek Tarun Garg

1. What: Problem Statement
Parameter-Efficient Fine-Tuning (PEFT) techniques like IA3, LoRA, FitBit, BOFT and Prefix Tuning etc have seen an increase in popularity, especially in the current context of scaling LLM architectures [Hu et al., 2022]. However, recent studies have indicated that when compared to full-finetuning, these methods converge more slowly in low-data scenarios. Our work aims to investigate if techniques such as conscious learning-rate scheduling, better regularization techniques and optimizer selection can overcome this slow convergence problem with PEFT methods. 
We aim to perform the following:
Evaluating the convergence patterns across different transformer architectures namely LLaMA, Mistral and FLAN-T5
Testing if changing the various hyperparameters like learning rate warmup, per-layer learning rate scheduler and optimizer selection can lead to faster convergence for classification and generation tasks even in low-data scenarios.



2. So What: Significance
PEFT is known for its memory efficiency, but longer training times due to their slow convergence in low-data scenarios highlights a bottleneck. PEFT techniques may become practically infeasible, especially when working in low-resource experimental setups with lesser data. Our work is strongly motivated by the following three factors:
Faster convergence will save computational resources enabling efficient and faster training given limited data. This could especially benefit small research labs and student projects who have limited GPU compute.
We plan to test multiple model architectures for our hypothesis to ensure generalizability of our experiments. If our findings are positive, then this generalizability could help ensure that these can be applied across various LLM architectures in the industry.
Our work will extend the existing research on PEFT techniques and conclude whether the slow convergence is an actual limitation or just an effect of inefficient hyperparameter tuning.

3. Now What: Proposed Methodology
3.1 Datasets & Resource Settings
 We will use the same datasets as the main reference paper [Pu et al., 2023] to ensure comparability:
Classification Tasks -
AG News: This is a dataset of around 127K News articles spanning across 4 domains: Business, World, Sport and Sci/Tech. Every sample in the dataset namely has 3 fields: class index, news article title and a brief summary of the article. This dataset is publicly available in HuggingFace and is widely used for natural language processing research.
CoLA: This dataset contains 10K sentences where each is labeled correct or incorrect   grammatically. The sentences have been gathered from published linguistics and manually annotated by experts.
Generation Tasks:
SAMSum: This dataset consists of around 15K made-up chat conversations which was developed with an aim to train AI systems to summarize chat conversations. Every entry in the dataset will contain a chat conversation and a human written summary of the chat.
E2E: This is a dataset that contains around 50K samples to train systems to generate descriptions of a restaurant. Every entry in the dataset contains restaurant details like name, price, type of food etc and another column contains a human description of that restaurant.
3.2 Models
 To test generalizability, we will compare the following models under 7B params (Touvron et al., 2023; Jiang et al., 2023; Wei et al., 2021):
LLaMA 
Mistral AI
FLAN
3.3 PEFT Techniques
We mainly plan to concentrate our work on IA3, LoRA, FitBit, and BOFT, to make sure that we have a good representative set of PEFT methods that alter different parts of the LLM. 
3.4 Experimental Setup
Optimizer & Learning Schedule Variations:
Optimizers: AdamW, Adafactor, Lion, etc.
Schedules: Constant LR, warmup schedules, layer-wise schedules, progressive or “curriculum” warmup.
Convergence Measurements: We will track time-to-convergence (epochs or steps to best validation loss), final accuracy (classification), and ROUGE-based metrics (summarization).
Resource Monitoring: Memory analysis, Time for training, Any Instability Instances like Gradient Explosions etc
Analysis: We will generate convergence curves (loss vs. training steps) for each (Model–PEFT–Optimizer–Schedule) combination under each resource setting.
4. Related Work & Background
PEFT Techniques: All the techniques: LoRA, IA3, FitBit, and BOFT reduce trainable parameters, but use different underlying design choices to do this. They ultimately lead to significant memory savings but with different design choices (adapters vs. bias-only updates vs. scaling parameters) [Hu et al., 2021; Liu et al., 2022].
Empirical Comparisons: A study by Scale AI recently found that PEFT methods can converge slower than full fine-tuning in small-data regimes, suggesting potential instability or suboptimal hyperparameter use. [Pu et al., 2023]. Our work builds on their findings and provides a more rigorous comparison to full-tuning vs PEFT.
Low-Resource Fine-Tuning: Prior work on prompt tuning, prefix tuning, and adapters underscores the difficulty of stable training when data is scarce [Lester et al., 2021; Ben Zaken et al., 2021].
Cross-Architecture Validation: FLAN, LLaMA, and Mistral differ in their pre-training techniques and attention mechanisms, which could possibly impact how quickly every method adapts with minimal data.

Additional Considerations
Uniqueness
While existing studies have benchmarked PEFT methods on performance, very few have systematically studied their convergence time under different optimizers and schedules. This makes their conclusions less applicable. We aim to address this gap in literature, and focus specifically on how training hyperparameters can mitigate or worsen slow convergence.
Evaluation
Quantitative: Steps or epochs to best validation loss (convergence speed), final accuracy (AG News, CoLA) or ROUGE (E2E, SAMSum).
Qualitative: Analyze generated text for summarization tasks under different hyperparameter regimes.
Secondary Metrics: Memory footprint, training time in GPU hours, number of NaN/infinite gradients.
Risks/Roadblocks
Compute Budget: The large grid search (multiple models × multiple optimizers × multiple schedules) can be time-intensive.
Mitigation: Start with smaller versions of the models or smaller subsets of data, then scale up once we see interesting trends. 
Hyperparameter Explosion: A broad search might yield many combinations.
Mitigation: Use adaptive search methods (e.g., Bayesian optimization) rather than exhaustive grids.
Overfitting in Low-Resource: Models may converge “instantly” to random noise if hyperparameters are mis-tuned.
Mitigation: Use careful early stopping and cross-validation strategies.
Conclusion
This project will explore whether PEFT’s slower convergence in low-data scenarios can be addressed by using tailored optimization strategies. By rigorously evaluating a range of PEFT techniques like IA3, LoRA, FitBit, and BOFT applied across multiple LLM architectures, we aim to deliver concrete guidelines on how to select optimizers, learning schedules, and hyperparameters to achieve both fast convergence and strong performance—especially under resource constraints. The findings will be valuable both for academic inquiry (clarifying an under-explored aspect of PEFT) and for industry practitioners looking to train models efficiently on small data.
"""

# given the above description, create a baseline peft training script.
# make it very simple and easy to understand.
# ensure it works for the above mentioned models and datasets.
# ensure you can log all the metrics mentioned in the description.
# mainly convergence speed, final accuracy, memory footprint, training time, any instability instances like gradient explosions etc.
# use wandb to log the metrics.
# use latest version of everything + all the best practices.





