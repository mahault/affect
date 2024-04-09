# emotionalinference

Emotional Inference Demonstration Overview
This demonstration, crafted with the collaboration of Riddhi Pitliya Jain, showcases an innovative approach to understanding emotional responses through active inference. It revolves around a scenario where we observe an agent's emotional states as they navigate through the challenge of locating their wallet within their home. This setup serves as a practical application for our emotional inference model, a sophisticated hierarchical active inference framework. At its core, the model integrates a dual-level analysis: the primary level focuses on the agent's task-specific actions (in this case, searching for a wallet), while the superior level reflecting its hierarchical nature, is tasked with the emotional analysis of the activities at the lower tier.

Additionally, the demonstration includes a control scenario—designated as the wallet-finding task devoid of the emotional inference architecture. This setup is intended to provide a comparative analysis against the comprehensive emotional-wallet-finding task, allowing for a clearer understanding of the emotional inference model's impact and effectiveness.

# To run experiments
To initiate the experimental process, specific preparatory steps are required:

1. Begin by purging the "experiments" folder of any existing files and removing the "results.txt" file to ensure a clean slate for new experimental data.

2. Navigate to the "config.yaml" file and modify the parameters according to the needs of the upcoming experiments. This step is crucial for tailoring the experimental conditions.

3. With the preparations complete, proceed to the directory containing your experimental files within the code terminal. Execute the command ./sweep.sh config.yaml 30, substituting "30" with the desired number of experiments to conduct in a single batch. This command initiates the automated experimental process.

4. As experiments are conducted, individual results will be systematically stored within the "experiments" folder. Upon completion of the batch, a consolidated compilation of results will be generated and saved in the "results.txt" file, facilitating an organized analysis of outcomes.

