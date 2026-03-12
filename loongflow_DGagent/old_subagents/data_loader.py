class DatasetLoader():
    """
        the workflow that loads the out source dataset to local.
        The main function of this workflow is to load the dataset and generate a cursory description for the dataset
        If a dataset is already loaded from the outside, we should tell users here.
    """
    def __init__(self,
                 name,
                 reasoning_model,
                 interacting_model,
                 ) -> None:
        self.name = name
        self.reasoning_model = reasoning_model
        self.interacting_model = interacting_model
        self.workflow = self._get_workflow()

    async def check_dataset(self, state: DataLoaderInputState) -> DataLoaderState:
        logger.info('----------DatasetLoader subgraph----------')
        new_dataset = DatasetInformation(
            dataset_name = state.dataset_name,
            input_dataset_path = state.input_dataset_path,
            dataset_user_description = state.dataset_user_description,
            workspace_path = state.workspace_path
        )

        return  DataLoaderState(dataset=new_dataset,
                                 **state.model_dump())

    async def dataset_loader(self, state: DataLoaderState):
        """
            Load (download) the dataset and save it in a given location.
        """
        logger.info(f"[Getting Dataset] The agent is trying to get the {state.dataset_name} dataset")
        try:
            shutil.copytree(state.input_dataset_path, state.dataset.input_data_save_dir_path,dirs_exist_ok=True)
            result = {"success": True, "ErrorMessage": ""}
            logger.info(f"Successfully get the {state.dataset_name} dataset")
        except Exception as e:
            error_message = f"Failed to get the {state.dataset_name} dataset, because {e}"
            logger.error(error_message)
            raise ValueError(error_message)
        logger.info(f"[Describing Dataset] The agent trying to describe the {state.dataset_name} dataset")
        dir_description_prompt = '''
# Tasks
Please generate a **One-sentence** description of the subfolder '{dir_path}' under the local dataset path '{dateset_path}' based on the following information.
# References
- Dataset path: '{dateset_path}'
- Subfolder path: '{dir_path}'
- User description of the entire dataset: "{dataset_user_description}"

# Output requirements
- Describe the role or content of the subfolder
- Describe the types of files that may be included in the subfolder
- If there are multiple subfolders, explain the relationship between them
- Concise and accurate language, suitable for dataset documentation or metadata descriptions
'''

        file_description_prompt = '''
# Tasks
Please generate a **One-sentence** description of the subfile '{file_path}' under the local dataset path '{dateset_path}' based on the information below.

# References
- Dataset path: '{dateset_path}'
- Subfile path: '{file_path}'
- User description of the entire dataset: "{dataset_user_description}"
- Description of the previous directory: "{dir_description}"

# Output requirements
- A general description of the contents of the document, including:
    - Whether it is a label file
    - Explain the role of the file in the dataset
    - Concise and accurate language, suitable for dataset documentation or metadata descriptions
'''

        path = Path(state.dataset.input_data_save_dir_path)

        files = {}
        dirs = {}

        keywords = {'dateset_path': state.dataset.input_data_save_dir_path,
                    'dataset_user_description': state.dataset_user_description}

        for item in path.iterdir():
            if item.is_dir():
                keywords.update({'dir_path': item})
                dir_response = self.reasoning_model.invoke(dir_description_prompt.format(**keywords))
                dir_description = dir_response.content
                dirs.update({item.name: dir_description})


        for item in path.iterdir():
            if item.is_file() and item.parent != path:
                dir_description = dirs.get(item.parent.name, "no folder description found")
                keywords.update({'file_path': item, 'dir_description': dir_description})
                file_response = self.reasoning_model.invoke(file_description_prompt.format(**keywords))
                file_description = file_response.content
                files.update({item.name: file_description})

            if item.is_file() and item.parent == path:
                dir_description = 'The previous directory is the directory where the dataset is located, so the description is the same as above'
                keywords.update({'file_path': item, 'dir_description': dir_description})
                file_response = self.reasoning_model.invoke(file_description_prompt.format(**keywords))
                file_description = file_response.content
                files.update({item.name: file_description})

        state.dataset.input_data_save_dir.sub_folders.update(dirs)
        state.dataset.input_data_save_dir.sub_files.update(files)

        return DataLoaderOutputState(
            success=result["success"],
            ErrorMessage=result["ErrorMessage"],
            dataset= state.dataset
        )


    def _get_workflow(self):

        # graph_builder
        graph_builder = StateGraph(
            input_schema = DataLoaderInputState,
            state_schema = DataLoaderState,
            output_schema = DataLoaderOutputState,
        )

        # nodes
        graph_builder.add_node("dataset_checker", self.check_dataset)
        graph_builder.add_node("dataset_loader", self.dataset_loader)

        # edges
        graph_builder.add_edge(START, "dataset_checker")
        graph_builder.add_edge("dataset_checker", "dataset_loader")
        graph_builder.add_edge("dataset_loader", END)

        dataloader_graph = graph_builder.compile(name = self.name)

        return dataloader_graph
