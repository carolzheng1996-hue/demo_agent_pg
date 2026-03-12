class DatasetFormatter():
    """
        Data normalization & description workflow,
        This workflow attempt to normalize the dataset based on user data and provide corresponding code for the user's dataset.
        Provide an initial description of the dataset for the user (based on the results after normalization).
        This will be used to detect whether the user's understanding aligns with that of the large model.
        TODO: In the future, this process should continue in a loop, allowing users to make preliminary modifications to the dataset until they achieve a satisfactory or desired dataset.
        After that, the dataset will be fixed for training and other operations.
    """

    def __init__(self,
                 name,
                 reasoning_model,
                 interacting_model,
                 langfuse_handler=None
                 ) -> None:
        self.name = name
        self.reasoning_model = reasoning_model
        self.interacting_model = interacting_model
        self.langfuse_handler = langfuse_handler
        self.workflow = self._build_workflow()
        dataset_tools = [PythonREPLTool(handle_tool_error=True)]
        self.dataset_process_agent = create_react_agent(model=reasoning_model,
                                                        tools=dataset_tools,
                                                        prompt="You are an assistant who helps people to load and process time series datasets.You are provided with python tools to execute the generated codes and help users process the dataset!!")


    def _find_supported_files(self, path):
        """reading and analyze data"""
        # 获取文件路径
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        valid_extensions = ['.csv', '.pkl', '.npy']
        data_files = [
            f for f in file_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        if len(data_files)==0:
            raise ValueError(f"Unsupported file type. Supported types: csv, pkl, npy")
        else:
            found_files = []
            for ext in valid_extensions:
                found_files.extend(list(file_path.glob(f"*{ext}")))

        all_files = list(set(found_files))
        all_files.sort()

        if len(all_files) > 100:
            logger.warning(f"Found {len(all_files)} files, limiting to 100")
            all_files = all_files[:100]
        return all_files

    async def _read_and_analyze_multiple_files(self, state):
        logger.info('----------DataReader Agent----------')

        # 设置默认值
        file_types = ['csv', 'pkl', 'npy']
        # 查找文件
        logger.info(f"🔍 Searching for files in: {state.dataset.input_data_save_dir.path}")
        logger.info(f"   - Supported types: {file_types}")
        logger.info(f"   - Max files: 100")

        supported_files = self._find_supported_files(state.dataset.input_data_save_dir.path)

        if not supported_files:
            raise FileNotFoundError(f"No supported files found in {state.dataset.input_data_save_dir.path}")

        logger.info(f"✅ Found {len(supported_files)} supported file(s):")
        for i, file_path in enumerate(supported_files[:5], 1):
            logger.info(f"   {i}. {file_path}")
        if len(supported_files) > 10:
            logger.info(f"   ... and {len(supported_files) - 10} more")

        # 读取1个文件作为示意
        logger.info("🔍 Reading one file as example...")
        file_path = supported_files[0]
        file_type = file_path.suffix.lower()
        dataset_info = {}
        try:
            if file_type == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"✅ CSV Data Info:")
                logger.info(f"   - Shape: {df.shape}")
                logger.info(f"   - Columns: {df.columns.tolist()}")
                logger.info(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
                dataset_info['type'] = 'csv'
                dataset_info['Shape'] = df.shape

            elif file_type == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                dataset_info['type'] = type(data).__name__

                if isinstance(data, pd.DataFrame):
                    logger.info(f"✅ PKL Data (DataFrame) Info:")
                    logger.info(f"   - Shape: {data.shape}")
                    logger.info(f"   - Columns: {data.columns.tolist()}")
                    dataset_info['Shape'] = data.shape
                elif isinstance(data, np.ndarray):
                    logger.info(f"✅ PKL Data (NumPy Array) Info:")
                    logger.info(f"   - Shape: {data.shape}")
                    logger.info(f"   - Dtype: {data.dtype}")
                    dataset_info['Shape'] = data.shape
                elif isinstance(data, dict):
                    dataset_info['keys'] = list(data.keys())
                    dataset_info['sample_items'] = {k: type(v).__name__ for k, v in list(data.items())[:5]}
                    logger.info(f"✅ PKL Data (Dictionary) Info:")
                    logger.info(f"   - Keys: {list(data.keys())}")
                else:
                    dataset_info['info'] = f"Object of type {type(data).__name__}"
                    logger.info(f"✅ PKL Data Info: {type(data).__name__}")

            elif file_type == '.npy':
                data = np.load(file_path, allow_pickle=True)
                logger.info(f"✅ NPY Data Info:")
                logger.info(f"   - Shape: {data.shape}")
                logger.info(f"   - Dtype: {data.dtype}")
                dataset_info['Shape'] = data.shape

            dataset_info['file_type'] = file_type

        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            raise RuntimeError(f"❌ Error reading file: {e}")
        state.dataset.valid_file = [str(i) for i in supported_files]
        state.dataset.dataset_info = dataset_info
        return state

    async def _ask_processing_preference(self, state):
        """Ask user preference"""

        logger.info("🤖: Would you like to provide custom data processing steps?")
        logger.info("Options:")
        logger.info("1. Yes, I'll provide custom processing steps")
        logger.info("2. No, use default processing")

        while True:
            try:
                logger.info("Enter your choice (1 or 2): ")
                choice = input().strip()
                if choice == '1':
                    use_custom_steps = True
                    logger.info("✅ You chose to provide custom processing steps.")
                    break
                elif choice == '2':
                    use_custom_steps = False
                    logger.info("✅ You chose to use default processing.")
                    break
                else:
                    logger.info("❌ Invalid choice. Please enter 1 or 2.")
            except Exception as e:
                logger.error(f"❌ Error: {e}")

        # 如果用户选择自定义步骤，获取处理步骤
        user_steps = " "
        if use_custom_steps:
            logger.info("📝 Please enter your data processing steps(Describe what you want to do with the data):")
            logger.info("Please note that the final output should include six files (x_train.npy,x_test.npy,x_valid.npy,y_train.npy,y_test.npy,y_valid.npy ),  If any file is missing, simply specify that file as empty.")
            user_steps = input().strip()
            logger.info(f"Received steps: {user_steps}") ##如果用户处理完不是x_train.npy,六个文件应该如何处理
        # 获取数据集切分比例（如果用户选择默认处理）
        else:
            while True:
                try:
                    logger.info("Please enter the input sequence length (input_seq_len, e.g., 50): ")
                    input_seq_len = input().strip()
                    if not input_seq_len:
                        logger.info("Using default input_seq_len = 50")
                        input_seq_len = 50
                    else:
                        input_seq_len = int(input_seq_len)
                    if input_seq_len <= 0:
                        raise ValueError
                    break
                except ValueError:
                    logger.info("❌ Invalid input. Please enter a positive integer.")

            task_str = str(state.task) if state.task else ""
            output_seq_len = 1  # 默认值（回归/分类/无监督）
            time_increment_len = 1
            if state.task != TaskType.CLASSIFICATION.value:
                logger.info(f"💡 Detected forecasting/regression task: '{task_str}'. Please specify output horizon.")
                while True:
                    try:
                        logger.info("Please enter the output sequence length (e.g.,10): ")
                        out_len = input().strip()
                        if not out_len:
                            logger.info("Using default output_seq_len = 1")
                            output_seq_len = 1
                        else:
                            output_seq_len = int(out_len)
                        if output_seq_len <= 0:
                            raise ValueError
                        break
                    except ValueError:
                        logger.info("❌ Invalid input. Please enter a positive integer.")

                while True:
                    try:
                        logger.info("Please enter the overlap step to apply sliding window to the dataset (e.g., 10): ")
                        time_increment = input().strip()
                        if not time_increment:
                            logger.info("Using the default overlap step = 1")
                            time_increment_len = 1
                        else:
                            time_increment_len = int(time_increment)
                        if time_increment_len < 1:
                            raise ValueError
                        break
                    except ValueError:
                        logger.info("❌ Invalid input. Please enter a positive integer.")
            else:
                logger.info(
                    f"💡 Task '{state.task}'output_seq_len = 1 (scalar label or zero).")

            while True:
                try:  #to do考虑直接推理的场景，不需要进行模型训练，直接利用基础模型进行zero-shot
                    logger.info(f"🤖: Please input the dataset split ratio, train:validation:test=? e.g. 7:2:1")
                    split_rate = input()
                    logger.info(f"🤖: User input dataset split ratio: train:validation:test={split_rate}")
                    ratios = [int(item) for item in split_rate.strip().split(":")]
                    if len(ratios) != 3:
                        raise ValueError("Must have exactly 3 parts.")
                    total = sum(ratios)
                    if total == 0:
                        raise ValueError("Sum cannot be zero.")
                    train_ratio, val_ratio, test_ratio = [r / total for r in ratios]
                    break
                except (ValueError, Exception) as e:
                    logger.info(f"❌ Invalid ratio: {e}. Please try again.")

            state.dataset.input_len = input_seq_len
            state.dataset.output_len = output_seq_len
            state.dataset.time_increment_len = time_increment_len

            state.dataset.train_ratio = train_ratio
            state.dataset.val_ratio = val_ratio
            state.dataset.test_ratio = test_ratio

        state.dataset.user_steps = user_steps
        state.dataset.use_custom_steps = use_custom_steps

        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _process_data_custom(self, state: DatasetFormatterState):
        """使用用户自定义步骤处理数据"""
        logger.info("\n🔧 Processing data with custom steps...")
        partial_input = {
            "input_data_save_dir": state.dataset.input_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "dataset_info": state.dataset.dataset_info,
            "user_steps": state.dataset.user_steps,
            "dataset_user_description":state.dataset.dataset_user_description
        }
        processing_prompt = PromptTemplate(
            input_variables=['input_data_save_dir',"split_output_dir","dataset_info", "user_steps",'dataset_user_description'],
            partial_variables=partial_input,
            template="""You are a time series data processing expert. Generate a complete, self-contained Python script to process the data according to **User Dataset Processing Steps**.
            
            Data Information:
            - `{input_data_save_dir}`: Raw data directory path
            - `{split_output_dir}`: Processed data output path
            - Dataset info: {dataset_info}
            - Raw data description:`{dataset_user_description}`
            **User Dataset Processing Steps**
            '{user_steps}'
            Code Requirements:
            1. Import all required libraries first
            2. Load the data from the given file path
            3. Process the data according to user steps
            4. Use **absolute paths only**. 
            5. Never shuffle, normalize, or overwrite the original dataset file.
            6. Generate ONLY the Python code. No explanations, comments (except essential ones), or markdown."""
        )
        code_generation_chain = processing_prompt | self.reasoning_model | StrOutputParser() #检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})

        # 保存代码文件，确认是否需要直接让大模型保存成python文件
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)

        input_dir = state.dataset.input_data_save_dir.path
        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = str(processing_code.strip())

        logger.info(f"💾 Processing code saved to: {code_file_path}")

        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _confirm_code(self, state):
        logger.info(f"\n⚠️ Please review the generated code in {state.dataset.code_save_dir.path}.")
        logger.info("\n" + "=" * 50)
        max_reflexion_rounds = 3
        current_round = 0
        while current_round < max_reflexion_rounds:
            logger.info("CODE REVIEW MENU:")
            logger.info("=" * 50)
            logger.info("1. 'confirm' to execute the code")
            logger.info("2. 'modify' to request changes")
            logger.info("3. 'cancel' to stop processing")

            try:
                logger.info("Enter your choice (1 or 2 or 3): ")
                resp = input().strip().lower()
                if resp =='1':
                    break
                elif resp =='2':
                    current_round += 1
                    if current_round < max_reflexion_rounds:
                        state = await self._handle_modification(state)
                    else:
                        raise SystemExit("Exceeded the number of modification times,User rejected code. Exiting.")
                elif resp =='3':
                    raise SystemExit("User rejected code. Exiting.")
                else:
                    logger.info("❌ Invalid choice. Please enter 1 or 2 or 3.")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                raise SystemExit("Unexpected error during code confirmation.")
        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _handle_modification(self, state):
        logger.info("You can describe issues like: 'fix the bug in function X', 'improve performance', etc.(or press Enter to skip):")
        feedback = input().strip() or "The code needs improvement."
        ## reflection
        partial_input = {
            "dataset_user_description": state.dataset.dataset_user_description,
            "feedback":feedback,
            "original_code":state.dataset.code_response
        }
        REFLEXION_prompt = PromptTemplate(
            input_variables=['dataset_user_description', "feedback", "original_code"],
            partial_variables=partial_input,
            template="""You are an expert Python data engineer acting as a code reviewer.Analyze the python code based on user feedback and task description.
            Please provide a concise, technical critique: identify concrete issues (e.g., logic errors, style, missing handling) and suggest specific improvements.
            **Raw data description**:{dataset_user_description}
            **User Feedback**: {feedback}
            **Previously Generated Python Code**:{original_code}""".strip()
        )

        # Step 1: Generate critique
        reflect_chain = REFLEXION_prompt | self.reasoning_model | StrOutputParser()
        critique_resp = await reflect_chain.ainvoke({})
        critique = critique_resp.strip()
        logger.info(f"🔍 Reflexion Critique:\n{critique}")
        partial_input = {
            "dataset_user_description": state.dataset.dataset_user_description,
            "feedback": feedback,
            "original_code": state.dataset.code_response,
            "critique":critique
        }

        # Step 2: Regenerate code with critique
        CODE_REGEN_PROMPT = PromptTemplate(
            input_variables=['dataset_user_description', "feedback", "original_code","critique"],
            partial_variables=partial_input,
            template="""You are an expert data engineer. Rewrite the original Python code to address the critique below.
            Original Python code :{original_code}
            **Raw data description**:{dataset_user_description}
            **User Feedback**: {feedback}
            **User critique**:{critique}
            **Code requirements**:
            - Import all required libraries first and end with saving the files. No other text. 
            - Validate paths with `os.path.exists()`.  
            - Use **absolute paths only**. 
            **Output only the Python code. Do not include markdown, explanations, or print statements.**
            """
        )

        regen_chain = CODE_REGEN_PROMPT | self.reasoning_model | StrOutputParser()
        new_code_resp = await regen_chain.ainvoke({})
        new_code = new_code_resp.strip()

        # Step 3: Save updated code
        script_dir = Path(state.dataset.script_file)
        with open(script_dir, 'w',encoding='utf-8') as f:
            f.write(new_code)

        state.dataset.processing_code = new_code
        logger.info(f"✅ Code regenerated and saved to {str(script_dir)}")
        return state

    async def _execute_code(self, state):
        logger.info("🚀 Executing confirmed code...")
        if hasattr(state.dataset, 'script_path') and state.dataset.script_file:   #需检查
            script_path = Path(state.dataset.script_file)
            with open(script_path, 'r', encoding='utf-8') as f:
                code_to_execute = f.read()
            prompt = f"You have full access to the user's file system, including absolute paths like 'D:\\code\\...',NEVER scan directories or load extra files.Do NOT refuse to run code due to path concerns — the paths are valid in this context.Please execute this Python code use the PythonREPLTool tool :\n\n{code_to_execute}\n"
        elif hasattr(state.dataset, 'code_response') and state.dataset.code_response:
            code_to_execute = state.dataset.code_response
            prompt = f"You have full access to the user's file system, including absolute paths like 'D:\\code\\...',NEVER scan directories or load extra files.Do NOT refuse to run code due to path concerns — the paths are valid in this context.Please execute this Python code use the PythonREPLTool tool :\n\n{code_to_execute}\n"
        else:
            raise ValueError("Neither 'script_path' nor 'code_response' found in state.")
        msg = HumanMessage(content=prompt)
        try:
            response = await self.dataset_process_agent.ainvoke({"messages": [msg]})
            logger.debug(f"Full agent response: {response}")
            exec_result = response["messages"]
            if exec_result[-2].tool_call_id:
                error = None
                raw_output = exec_result[-1].content
            else:
                raw_output = None
                error = "Agent did not invoke code execution tool. Possible refusal or planning step."
        except Exception as e:
            logger.error(f"Execution failed during agent invocation: {e}")
            raw_output=None
            error= str(e)
        if error == None:
            error = ''
            success = True
        else:
            success = False
        logger.info(f"Execution result: success={success}, output={raw_output},error={error}")

        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task,
            success=success,
            ErrorMessage=error
        )

        async def _process_forcasting_data_default(self, state):
        """使用默认步骤处理数据"""
        logger.info("\n🔧 Splitting data with default steps...")
        partial_input = {
            "format_data_save_dir": state.dataset.format_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "train_ratio":state.dataset.train_ratio,
            "val_ratio":state.dataset.val_ratio,
            "test_ratio":state.dataset.test_ratio,
            "time_increment_len":state.dataset.time_increment_len,
            "input_len":state.dataset.input_len,
            "output_len":state.dataset.output_len
        }
        default_prompt = PromptTemplate(
            input_variables=["format_data_save_dir", "train_ratio", "val_ratio","test_ratio","time_increment_len", "input_len", "output_len","split_output_dir"],
            partial_variables=partial_input,
            template="""You are an expert in time-series data preprocessing. Generate **only Python code** (no explanations or comments beyond necessary docstrings) that performs the following steps exactly:
            1. **Load data**:  
                - Read all .npy data files from `{format_data_save_dir}`.
                - **Do NOT change** the original files in `{format_data_save_dir}`.
            2. **Sequential train/val/test split**:  
                - Split the data **sequentially** (no shuffling) into three segments using ratios: train:val:test={train_ratio} : {val_ratio}: {test_ratio}  
                - Calculate split points: int(total_len * ratio)
                - Contiguous segments: [0:train_end], [train_end:val_end], [val_end:]
            3. **Sliding window construction**:  
                - For each split (train, val, test), apply a sliding window with step size `{time_increment_len}` to generate samples.  
                - Implement logic equivalent to:
                 def construct_sliding_window_data(data, Input_length, output_length, time_increment={time_increment_len}):
                     n = len(data) - Input length - output length + 1
                     if n <= 0:
                         F = data.shape[1]
                         return np.zeros((0, F, Input_length), dtype=np.float32), np.zeros((0, F, output_length), dtype=np.float32)
                     idx = np.arange(0, n, time_increment)
                     X = np.stack([data[i:i+Input_length].T for i in idx], axis=0).astype(np.float32)
                     Y = np.stack([data[i+Input_length:i+Input_length+output_length].T for i in idx], axis=0).astype(np.float32)
                     return X, Y
                 - Input length = {input_len}, output length = {output_len}
                 - if n>0, ensure X.shape[0] and Y.shape[0] equal to len(np.arange(0, n, time_increment))
            4. **Save output files**:  
               - Save exactly six `.npy` files to `{split_output_dir}` with these **exact names**:  
                 - `X_train.npy`, `X_valid.npy`, `X_test.npy`  
                 - `Y_train.npy`, `Y_valid.npy`, `Y_test.npy`  
               - Ensure the shape of the final output files to be:
                 - `X_*.npy`: shape `[N, F, {input_len}]`, `N ≥ 0`
                 - `Y_*.npy`: shape `[N, F, {output_len}]`, `N ≥ 0`
            5. **Code requirements**:
               - Import all required libraries first and end with saving the files. No other text. 
               - Validate paths with `os.path.exists()`.  
               - Do not use random seed or shuffle.
               - Use **absolute paths only**. 
               - Split along sample num dimension without shuffling using the given ratios.
            **Output only the Python code. Do not include markdown, explanations, or print statements.**  
                 """
        )

        code_generation_chain = default_prompt | self.reasoning_model | StrOutputParser()  # 检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})
        if not processing_code:
            raise ValueError("Generated code is invalid or empty.")
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)
        input_dir = state.dataset.input_data_save_dir.path
        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_forecasting_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = processing_code.strip()
        logger.info(f"💾 Processing code saved to: {code_file_path}")

        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _process_classification_data_default(self, state):
        logger.info(f"🤖: Start classification datasets splitting ")
        partial_input = {
            "format_data_save_dir": state.dataset.format_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "train_ratio": state.dataset.train_ratio,
            "val_ratio": state.dataset.val_ratio,
            "test_ratio": state.dataset.test_ratio,
            "input_len": state.dataset.input_len
        }
        class_prompt_template = PromptTemplate(
            input_variables=["format_data_save_dir", "input_len", "train_ratio", "val_ratio", "test_ratio","split_output_dir"],
            partial_variables=partial_input,
            template="""You are an expert in time-series data preprocessing. Generate **only Python code** (no explanations or comments beyond necessary docstrings) that performs the following steps exactly:
            1. **Load data**:  
                - Read all .npy data files from `{format_data_save_dir}`.
                - **Do NOT change** the original files in `{format_data_save_dir}`.
            2. **Sequential train/val/test split**:  
                - Split the data **sequentially** (no shuffling) into three segments using ratios: train:val:test={train_ratio} : {val_ratio}: {test_ratio}  
                - Calculate split points: int(total_len * ratio)
                - Contiguous segments: [0:train_end], [train_end:val_end], [val_end:]
            3. **Save exactly 6 data files to `{split_output_dir}` use these exact names:**
                - X_train.npy, X_valid.npy, X_test.npy
                - Y_train.npy, Y_valid.npy, Y_test.npy
                - Ensure the shape of the final files to be:
                    - X_*.npy: shape [N, F, {input_len}], N ≥ 0
                    - Y_*.npy: shape [N, 1, 1], N ≥ 0 — **strictly three-dimensional with second and third dimensions equal to 1**
            4. **Code requirements**:
               - Import all required libraries first and end with saving the files. No other text. 
               - Validate paths with `os.path.exists()`.  
               - Handle empty or invalid splits gracefully (e.g., zero-sample arrays).  
               - Do not use random seed or shuffle.
               - Split along sample num dimension without shuffling using the given ratios.
               - Use **absolute paths only**. 
               - If the Y data is not already shaped as [N, 1, 1], explicitly reshape or expand dimensions to enforce [N, 1, 1] before saving..
            **Output only the Python code. Do not include markdown, explanations, or print statements.**""".strip()
        )
        code_generation_chain = class_prompt_template | self.reasoning_model | StrOutputParser()  # 检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})
        if not processing_code:
            raise ValueError("Generated code is invalid or empty.")
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)

        input_dir = state.dataset.input_data_save_dir.path

        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_cla_reg_ad_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)
        logger.info(f"💾 Processing code saved to: {code_file_path}")

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = processing_code.strip()
        logger.info(f"💾 Processing code saved to: {code_file_path}")
        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _process_anomaly_detection_data_default(self, state):
        logger.info(f"🤖: Start anomaly_detection splitting ")
        partial_input = {
            "format_data_save_dir": state.dataset.format_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "train_ratio": state.dataset.train_ratio,
            "val_ratio": state.dataset.val_ratio,
            "test_ratio": state.dataset.test_ratio,
            "input_len": state.dataset.input_len
        }
        class_prompt_template = PromptTemplate(
            input_variables=["format_data_save_dir", "input_len", "train_ratio", "val_ratio", "test_ratio","split_output_dir"],
            partial_variables=partial_input,
            template="""You are an expert in time-series data preprocessing. Generate **only Python code** (no explanations or comments beyond necessary docstrings) that performs the following steps exactly:
            1. **Load data**:  
                - Read all .npy data files from `{format_data_save_dir}`.
                - **Do NOT change** the original files in `{format_data_save_dir}`.
            2. **Sequential train/val/test split**:  
                - Split the data **sequentially** (no shuffling) into three segments using ratios: train:val:test={train_ratio} : {val_ratio}: {test_ratio}  
                - Calculate split points: int(total_len * ratio)
                - Contiguous segments: [0:train_end], [train_end:val_end], [val_end:]
            3. **Save exactly 6 data files to `{split_output_dir}` use these exact names:**
                - X_train.npy, X_valid.npy, X_test.npy
                - Y_train.npy, Y_valid.npy, Y_test.npy
                - Ensure the shape of the final files to be:
                 - `X_*.npy`: shape `[N, F, {input_len}]`, `N ≥ 0`
                 - `Y_*.npy`: shape `[N, F, 1]`, `N ≥ 0`
            4. **Code requirements**:
               - Import all required libraries first and end with saving the files. No other text. 
               - Validate paths with `os.path.exists()`.  
               - Handle empty or invalid splits gracefully (e.g., zero-sample arrays).  
               - Do not use random seed or shuffle.
               - Split along sample num dimension without shuffling using the given ratios.
               - Use **absolute paths only**. 
            **Output only the Python code. Do not include markdown, explanations, or print statements.**""".strip()
        )
        code_generation_chain = class_prompt_template | self.reasoning_model | StrOutputParser()  # 检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})
        if not processing_code:
            raise ValueError("Generated code is invalid or empty.")
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)

        input_dir = state.dataset.input_data_save_dir.path

        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_cla_reg_ad_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)
        logger.info(f"💾 Processing code saved to: {code_file_path}")

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = processing_code.strip()
        logger.info(f"💾 Processing code saved to: {code_file_path}")
        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _process_feature_extraction_task(self, state):
        logger.info(f"🤖: Start feature_extraction datasets splitting ")
        partial_input = {
            "format_data_save_dir": state.dataset.format_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "train_ratio": state.dataset.train_ratio,
            "val_ratio": state.dataset.val_ratio,
            "test_ratio": state.dataset.test_ratio,
            "input_len": state.dataset.input_len
        }
        class_prompt_template = PromptTemplate(
            input_variables=["format_data_save_dir", "input_len", "train_ratio", "val_ratio", "test_ratio","split_output_dir"],
            partial_variables=partial_input,
            template="""You are an expert in time-series data preprocessing. Generate **only Python code** (no explanations or comments beyond necessary docstrings) that performs the following steps exactly:
            1. **Load data**:  
                - Read all .npy data files from `{format_data_save_dir}`.
                - **Do NOT change** the original files in `{format_data_save_dir}`.
            2. **Sequential train/val/test split**:  
                - Split the data **sequentially** (no shuffling) into three segments using ratios: train:val:test={train_ratio} : {val_ratio}: {test_ratio}  
                - Calculate split points: int(total_len * ratio)
                - Contiguous segments: [0:train_end], [train_end:val_end], [val_end:]
            3. **Save exactly 6 data files to `{split_output_dir}` use these exact names:**
                - X_train.npy, X_valid.npy, X_test.npy
                - Y_train.npy, Y_valid.npy, Y_test.npy (all Y_*.npy filled with zero)
                - Ensure the shape of the final files to be:
                 - `X_*.npy`: shape `[N, F, {input_len}]`, `N ≥ 0`
                 - `Y_*.npy`: shape `[N, F, 1]`, `N = 0`
            4. **Code requirements**:
               - Import all required libraries first and end with saving the files. No other text. 
               - Validate paths with `os.path.exists()`.  
               - Handle empty or invalid splits gracefully (e.g., zero-sample arrays).  
               - Do not use random seed or shuffle.
               - Split along sample num dimension without shuffling using the given ratios.
               - Use **absolute paths only**. 
            **Output only the Python code. Do not include markdown, explanations, or print statements.**""".strip()
        )
        code_generation_chain = class_prompt_template | self.reasoning_model | StrOutputParser()  # 检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})
        if not processing_code:
            raise ValueError("Generated code is invalid or empty.")
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)

        input_dir = state.dataset.input_data_save_dir.path

        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_cla_reg_ad_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)
        logger.info(f"💾 Processing code saved to: {code_file_path}")

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = processing_code.strip()
        logger.info(f"💾 Processing code saved to: {code_file_path}")
        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
       async def _load_data_default(self, state):
        logger.info("\n🔧 Load and merge data with default steps...")
        partial_input = {
            "input_data_list": state.dataset.valid_file,
            "format_data_save_dir":state.dataset.format_data_save_dir.path,
            "split_output_dir": state.dataset.split_output_dir.path,
            "dataset_info": json.dumps(state.dataset.dataset_info, indent=2),
            "dataset_user_description":state.dataset.dataset_user_description,
            "train_ratio": state.dataset.train_ratio,
            "val_ratio": state.dataset.val_ratio,
            "test_ratio": state.dataset.test_ratio,
        }
        default_prompt = PromptTemplate(
            input_variables=["input_data_list", "dataset_info", "train_ratio", "val_ratio", "test_ratio",
                             "split_output_dir", "dataset_user_description"],
            partial_variables=partial_input,
            template="""You are an expert in time-series data preprocessing. Generate a complete, self-contained Python code to process the data according to **Default Dataset Processing Steps**.
                   Data Information:
                       - Dataset info: {dataset_info}
                       - Raw data description:`{dataset_user_description}`
                   **Default Dataset Processing Steps**:
                       1. **Load files**:  Read all files from the list `{input_data_list}`(a list of absolute file paths).
                       2. Group and merge files that follow the same naming pattern with numeric suffixes  
                           - Group files **only if** their filenames share a common prefix** (e.g., `X_1.csv`, `X_2.csv` → group as `X`; `feature_train_1.npy`, `feature_train_2.npy` → group as `feature_train`).  
                           - **Never group** files with different semantic roles (e.g., `X` vs `y`, `train` vs `label`, `features` vs `targets`).
                       3. **Concatenate within groups**:  For each group, sort files by filename and concatenate their contents **along the first (time) axis** into a NumPy array of dtype `float64`, preserving original order.   
                       4. **Save output files**:  Saved the processed data as .npy files in `{format_data_save_dir}` 
                   **Code requirements**:
                    - Import all required libraries first
                    - Load the data from the given file path
                    - Process the data according to user steps  
                    - Use **absolute paths only**. 
                    - Never normalize, or overwrite the original dataset file.
                    -Generate ONLY the Python code. No explanations, comments (except essential ones), or markdown.""".strip()
        )
        code_generation_chain = default_prompt | self.reasoning_model | StrOutputParser()  # 检查使用哪种形式
        processing_code = await code_generation_chain.ainvoke({})
        code_save_dir = Path(state.dataset.code_save_dir.path)
        code_save_dir.mkdir(exist_ok=True)

        input_dir = state.dataset.input_data_save_dir.path
        dir_basename = os.path.basename(input_dir.rstrip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        code_filename = f"process_load_{dir_basename}_{timestamp}.py"
        code_file_path = os.path.join(code_save_dir, code_filename)
        with open(code_file_path, 'w',encoding='utf-8') as f:
            f.write(processing_code)

        state.dataset.script_file = str(code_file_path)
        state.dataset.code_response = str(processing_code.strip())
        logger.info(f"💾 Data loading code saved to: {code_file_path}")
        return DatasetFormatterState(
            dataset=state.dataset,
            task=state.task
        )

    async def _process_data_default(self, state):
        logger.info("\n🔧 Splitting data with default steps...")
        if state.task == TaskType.FORECASTING.value:
            data_state = await self._process_forcasting_data_default(state)
        elif state.task == TaskType.ANOMALY_DETECTION.value:
            data_state = await self._process_anomaly_detection_data_default(state)
        elif state.task == TaskType.FEATURE_EXTRACTION.value:
            data_state = await self._process_feature_extraction_task(state)
        elif state.task == TaskType.CLASSIFICATION.value:
            data_state = await self._process_classification_data_default(state)
        else:
            raise ValueError("Unsupported task type.")

        return DatasetFormatterState(
            dataset=data_state.dataset,
            task=data_state.task
        )

    def _final_check(self, state):
        split_data_dir = state.dataset.split_output_dir_path
        x_train = np.load(Path(split_data_dir) / 'X_train.npy')
        y_train = np.load(Path(split_data_dir) / 'Y_train.npy')
        x_test = np.load(Path(split_data_dir) / 'X_test.npy')
        y_test = np.load(Path(split_data_dir) / 'Y_test.npy')
        x_valid = np.load(Path(split_data_dir) / 'X_valid.npy')
        y_valid = np.load(Path(split_data_dir) / 'Y_valid.npy')
        logger.info(f"split data `X_train.npy` shape: {x_train.shape}")
        logger.info(f"split data `Y_train.npy` shape: {y_train.shape}")
        logger.info(f"split data `X_test.npy` shape: {x_test.shape}")
        logger.info(f"split data `Y_test.npy` shape: {y_test.shape}")
        logger.info(f"split data `X_valid.npy` shape: {x_valid.shape}")
        logger.info(f"split data `Y_valid.npy` shape: {y_valid.shape}")
        logger.info(f"The data shape is correct or not? [y/n]")
        check = input()
        if check.lower() == 'y':
            return True
        else:
            return False


    def _build_workflow(self):
        # graph_builder
        graph_builder = StateGraph(
            input_schema=DatasetFormatterInputState,
            state_schema=DatasetFormatterState,
            output_schema=DatasetFormatterOutputState,
        )

        # nodes
        graph_builder.add_node("read_format", self._read_and_analyze_multiple_files)
        graph_builder.add_node("ask_preference", self._ask_processing_preference)
        graph_builder.add_node("process_custom", self._process_data_custom)
        graph_builder.add_node("_load_data_default", self._load_data_default)
        graph_builder.add_node("_process_data_default", self._process_data_default)

        graph_builder.add_node("confirm_custom_or_process", self._confirm_code)
        graph_builder.add_node("execute_custom_or_process", self._execute_code)
        graph_builder.add_node("confirm_load", self._confirm_code)
        graph_builder.add_node("execute_load", self._execute_code)
        graph_builder.add_node("final_check", self._final_check)

        def route_after_ask(state):
            return "process_custom" if state.dataset.use_custom_steps else "_load_data_default"

        # edges
        graph_builder.add_edge(START, "read_format")
        graph_builder.add_edge("read_format", "ask_preference")

        graph_builder.add_conditional_edges(
            "ask_preference",
            route_after_ask,
            {
                "process_custom": "process_custom",
                "_load_data_default": "_load_data_default"
            }
        )
        # Custom path
        graph_builder.add_edge("process_custom", "confirm_custom_or_process")
        graph_builder.add_edge("confirm_custom_or_process", "execute_custom_or_process")
        graph_builder.add_conditional_edges("execute_custom_or_process", self._final_check,
                                            {True: END, False: 'ask_preference'})

        # Default path
        graph_builder.add_edge("_load_data_default", "confirm_load")
        graph_builder.add_edge("confirm_load", "execute_load")
        graph_builder.add_edge("execute_load", "_process_data_default")
        graph_builder.add_edge("_process_data_default", "confirm_custom_or_process")  # reuse
        graph_builder.add_edge("confirm_custom_or_process", "execute_custom_or_process")


        # if self.langfuse_handler is None:
        dataloader_graph = graph_builder.compile(name=self.name)
        # else:
        #     dataloader_graph = graph_builder.compile(name=self.name).with_config({"callbacks": [self.langfuse_handler]})

        # dataloader_graph = graph_builder.compile(name = self.name)
        return dataloader_graph

