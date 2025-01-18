# LLM Benchmark Reproduction Results

| Benchmark   | Tester  | Model                                   | Metric/Score                  | Our Results                    | Reported Results              |
|-------------|---------|-----------------------------------------|--------------------------------|--------------------------------|-------------------------------|
| Alpaca Eval | Etash   | Meta LLama 3 8b Instruct               | win_rate                      | 24.04                          | 22.57                        |
|             |         |                                         | standard_error                | 1.27                           | 1.26                         |
|             |         |                                         | avg_length                    | 1937                           | 1899                         |
|             |         |                                         | length_controlled_winrate     | 24.24                          | 22.92                        |
| HumanEval   | Etash   | DeepSeek Coder 6.7B Instruct           | Python                        | 0.798                          | 78.9%                        |
|             |         |                                         | Bash                          | 0.354                          | 36.7%                        |
| MBPP        | Etash   | DeepSeek Coder 6.7B Instruct           | pass@1                        | 0.64                           | 65.4%                        |
| RepoBench   | Negin   | StarCoder                              | EM (cross_file_first)         | 28.0                           | 28.0                         |
|             |         |                                         | ES (cross_file_first)         | 67.37                          | 69.6                         |
|             |         |                                         | EM (cross_file_random)        | 37.08                          | 37.3                         |
|             |         |                                         | ES (cross_file_random)        | 71.28                          | 73.69                        |
|             |         |                                         | EM (in_file)                  | 34.17                          | 33.8                         |
|             |         |                                         | ES (in_file)                  | 70.46                          | 72.37                        |
|             |         |                                         | EM (weighted avg)             | 31.69                          | 31.69                        |
|             |         |                                         | ES (weighted avg)             | 69.09                          | 71.2                         |
|             |         | Codegen-350M-mono                      | EM (cross_file_first)         | 15.27                          | 15.14                        |
|             |         |                                         | ES (cross_file_first)         | 58.03                          | 60.1                         |
|             |         |                                         | EM (cross_file_random)        | 27.7                           | 27.7                         |
|             |         |                                         | ES (cross_file_random)        | 67.33                          | 68.9                         |
|             |         |                                         | EM (in_file)                  | 25.11                          | 25.2                         |
|             |         |                                         | ES (in_file)                  | 66.28                          | 67.8                         |
|             |         |                                         | EM (weighted avg)             | 22.12                          | 20.7                         |
|             |         |                                         | ES (weighted avg)             | 62.9                           | 64.2                         |
|             |         | Codegen-2B-mono                        | EM (cross_file_first)         | 22.12                          | 22.1                         |
|             |         |                                         | ES (cross_file_first)         | 62.9                           | 64.9                         |
|             |         |                                         | EM (cross_file_random)        | 34.18                          | 34.4                         |
|             |         |                                         | ES (cross_file_random)        | 71.12                          | 72.6                         |
|             |         |                                         | EM (in_file)                  | 31.14                          | 31.2                         |
|             |         |                                         | ES (in_file)                  | 69.61                          | 70.93                        |
|             |         |                                         | EM (weighted avg)             | 27.26                          | 27.3                         |
|             |         |                                         | ES (weighted avg)             | 66.57                          | 68.3                         |
|             |         | Codegen-6B-mono                        | EM (cross_file_first)         | 26.81                          | 26.9                         |
|             |         |                                         | ES (cross_file_first)         | 66.23                          | 67.9                         |
|             |         |                                         | EM (cross_file_random)        | 38.28                          | 38.3                         |
|             |         |                                         | ES (cross_file_random)        | 73.34                          | 74.5                         |
|             |         |                                         | EM (in_file)                  | 34.9                           | 34.96                        |
|             |         |                                         | ES (in_file)                  | 71.21                          | 72.59                        |
|             |         |                                         | EM (weighted avg)             | 31.56                          | 31.67                        |
|             |         |                                         | ES (weighted avg)             | 69.16                          | 70.68                        |
| MTBench     | Etash   | stabilityai/stablelm-tuned-alpha-7b    |                               | 1.0                            | 2.75                         |
|             |         | Nexusflow/Starling-LM-7B-beta          |                               | 7.69                           | 8.12                         |
|             |         | mistralai/Mistral-7B-Instruct-v0.1     |                               | 6.65                           | 6.84                         |
|             |         | databricks/dolly-v2-12b                |                               | 1.087                          | 3.28                         |
| WildBench   | Etash   | princeton-nlp/gemma-2-9b-it-SimPO      |                               | 5.083                          | 5.33                         |
|             |         | meta-llama/Meta-Llama-3-8B-Instruct    |                               | 3.27                           | 2.92                         |
| IFEval      | Negin   | meta-llama/Llama-3.1-8B-Instruct       |                               | 79.1                           | 80.4                         |
| ZeroEval    | Negin   | meta-llama/Llama-3.1-8B-Instruct       |crux                           | 40.75                          | 39.88
|             |         |                                        |math-l5                        | 24.69                          | 22.19
|             |         |                                        |zebra                          | 11.70                          | 12.8