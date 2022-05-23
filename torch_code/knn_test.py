import seaborn
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# from torch_code.torch_tests import TorchExample, parse_args, DataTuple
from torch_tests import TorchExample, parse_args, DataTuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from tabulate import tabulate
from numpy import linalg as la

print("The print that I forgot to delete")


def disply_param_histogram(model, model_type):
    for param_name, param in model.params.items():
        if 'W' in param_name:
            norm = la.norm(param)
            seaborn.histplot(param)  # , label=f"{param_name}, {reg_strength}")
            plt.title(f"{param_name}, reg: {model.reg}, {model_type}, norm: {norm}")
            plt.show()


class KNNTest(TorchExample):
    def __init__(self, args, task):
        TorchExample.__init__(self, args=args, task=task)

    def get_feature_extractor(self, model):
        # model_ft = models.resnet18(pretrained=True)
        ### strip the last layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        return feature_extractor
        ### check this works
        # x = torch.randn([1,3,224,224])
        # output = feature_extractor(x) # output now has the features corresponding to input x
        # print(output.shape)

    def knn_prediction(self, model: torch.nn.Sequential, datatuple: DataTuple):
        feature_extractor = self.get_feature_extractor(model)
        train_knn_x, train_knn_y = self.get_features(feature_extractor, datatuple.train)
        val_knn_x, val_knn_y = self.get_features(feature_extractor, datatuple.val)
        neigh = KNeighborsClassifier(n_neighbors=10)
        train_knn_x = np.reshape(train_knn_x, (train_knn_x.shape[0], -1))
        neigh.fit(train_knn_x, train_knn_y)
        train_predictions = neigh.predict(train_knn_x)
        train_accuracy = accuracy_score(train_predictions, train_knn_y)
        val_knn_x = np.reshape(val_knn_x, (val_knn_x.shape[0], -1))
        prediction = neigh.predict(val_knn_x)
        accuracy = accuracy_score(prediction, val_knn_y)
        return accuracy, train_accuracy
        # print(neigh.predict([[1.1]]))
        # print(neigh.predict_proba([[0.9]]))

    def get_features(self, model, dataloader):
        features = []
        labels = []
        for x, y in dataloader:
            x = x.to(device=self.device, dtype=torch.float32)
            features.append(model(x).cpu().detach().numpy())
            labels.append(y.detach().numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        return features, labels

    def create_dataloaders_subsets(self):
        first_cifar_subset = self.get_pytorch_cifar_data()[
            0]._asdict()  # CIFARSubset(subsets["first split data"], subsets["first split labels"])
        second_cifar_subset = self.get_pytorch_cifar_data()[
            0]._asdict()  # CIFARSubset(subsets["second split data"], subsets["second split labels"])
        for key, subset in self.data._asdict().items():
            subsets = self.divide_to_sub_sets(subset.dataset)

            first_cifar_subset[key].data = subsets["first split data"]
            first_cifar_subset[key].targets = subsets["first split labels"]
            second_cifar_subset[key].data = subsets["second split data"]
            second_cifar_subset[key].targets = subsets["second split labels"]
            # first_subset_dict[key] = first_cifar_subset
            # second_subset_dict[key] = second_cifar_subset
        first_datatuple = DataTuple(**first_cifar_subset)
        second_datatuple = DataTuple(**second_cifar_subset)
        return first_datatuple, second_datatuple

    def knn_experiment(self):
        knn_df = {}
        first_datatuple, second_datatuple = self.create_dataloaders_subsets()

        self.data = first_datatuple  # self.get_data_loaders(datatuple=first_datatuple)
        # train
        result_df, original_model, adaptive_model = self.train_and_eval()
        # knn results
        print(tabulate(result_df))
        knn_df["Adaptive model"] = self.knn_prediction(adaptive_model, second_datatuple)
        knn_df["Regular model"] = self.knn_prediction(original_model, second_datatuple)
        # knn_df = {
        #     "original_val_accuracy": original_accuracy,
        #     "original_train_accuracy": original_train_accuracy,
        #     "adaptive_val_accuracy": adaptive_accuracy,
        #     "adaptive_train_accuracy": adaptive_train_accuracy
        # }
        print(knn_df)
        knn_df = pd.DataFrame(knn_df)
        knn_df = knn_df.transpose()
        knn_df.columns = ["Val KNN acc", "Train KNN acc"]
        if self.logger:
            self.logger.report_table(title="KNN Accuracy", series="KNN Accuracy",
                                     iteration=0, table_plot=knn_df)
        return result_df, knn_df

    def mean_and_ci_result(self):
        # if self.args.trains:
        #     task = Task.init(project_name='Flexible Regularization',
        #                      task_name='train_and_eval')  # , reuse_last_task_id=False)
        # else:
        #     task = None
        acc_tables, knn_tables = [], []
        for _ in range(self.args.num_of_repeats):
            results_df, knn_df = self.knn_experiment()
            acc_tables.append(results_df)
            knn_tables.append(knn_df)
        print("experiments results")
        self.report_experiment_tables(acc_tables, table_name='experiment classes acc')
        print("KNN results")
        self.report_experiment_tables(knn_tables, table_name='KNN classes acc')

    # def report_acc_experiment_tables(self, tables):
    #     import pandas as pd
    #     # content = [df.drop(columns=['Optimizer', 'Adaptive?']).values for df in tables]
    #     tables_val = np.asarray([pd.values for pd in tables])
    #     mean_vales = np.mean(tables_val, axis=0)
    #     var_values = np.var(tables_val, axis=0)
    #     mean_df = pd.DataFrame(mean_vales, columns=tables[0].columns)
    #     var_df = pd.DataFrame(var_values, columns=tables[0].columns)
    #     print("mean_df")
    #     print(mean_df)
    #     print("var_df")
    #     print(var_df)
    #     if self.args.trains:
    #         task.get_logger().report_table(title='Mean values', series='Mean values',
    #                                        iteration=self.args.num_trains, table_plot=mean_df)
    #
    #         task.get_logger().report_table(title='Mean values', series='Mean values',
    #                                        iteration=self.args.num_trains, table_plot=var_df)

    def report_experiment_tables(self, tables, table_name):
        import pandas as pd
        # content = [df.drop(columns=['Optimizer', 'Adaptive?']).values for df in tables]
        tables_val = np.asarray([pd.values for pd in tables])
        mean_vales = np.mean(tables_val, axis=0)
        var_values = np.var(tables_val, axis=0)
        mean_df = pd.DataFrame(mean_vales, columns=tables[0].columns)
        var_df = pd.DataFrame(var_values, columns=tables[0].columns)
        # mean_df.columns = ["val acc", "train acc"]
        mean_df.index = ["Adaptive model", "Regular model"]
        mean_df = mean_df.transpose()

        var_df.index = ["Adaptive model", "Regular model"]
        var_df = var_df.transpose()

        # var_df.columns = ["val acc", "train acc"]
        print("mean df")
        print(tabulate(mean_df, headers=mean_df.columns))
        print(mean_df)
        print("var df")
        print(tabulate(var_df, headers=var_df.columns))
        if self.logger:
            # logger = task.get_logger()
            self.logger.report_table(title=f'{table_name} mean values', series=f'{table_name} mean values',
                                           iteration=self.args.num_trains, table_plot=mean_df)

            self.logger.report_table(title=f'{table_name} var', series=f'{table_name} var',
                                           iteration=self.args.num_trains, table_plot=var_df)
        else:
            mean_df.to_csv(f"{self.args.output_dir}/{table_name}.csv")
            var_df.to_csv(f"{table_name}.csv")

        # stacked_content = np.stack(tables)
        # mean_values = pd.DataFrame(np.mean(stacked_content, axis=0))
        # std = pd.DataFrame(np.std(stacked_content, axis=0))
        # print(mean_values)
        # print(std)
        # second_column, third_column = tables[0]['Optimizer'], tables[0]['Adaptive?']
        # mean_values.insert(loc=2, column='Optimizer', value=second_column)
        # mean_values.insert(loc=3, column='Adaptive', value=third_column)
        # mean_values.columns = tables[0].columns
        # std.insert(loc=2, column='Optimizer', value=second_column)
        # std.insert(loc=3, column='Adaptive', value=third_column)
        # std.columns = tables[0].columns
        # print("avg values")
        # print(tabulate(mean_values, headers=mean_values.columns))
        # if self.args.trains:
        #     task.get_logger().report_table(title='Mean values', series='Mean values',
        #                                    iteration=self.args.num_trains, table_plot=mean_values)
        # print("standard deviation")
        # print(tabulate(std, headers=std.columns))
        # if self.args.trains:
        #     task.get_logger().report_table(title='Standard deviation', series='Standard deviation',
        #                                    iteration=self.args.num_trains, table_plot=std)


def main():
    args = parse_args()
    if args.trains:
        from trains import Task
        task = Task.init(project_name='Flexible Regularization',
                         task_name='KNN, 5 seen classes reg 0.1 resnet18 1000 trains 20 epoch')
    else:
        task = None
    torch_example = KNNTest(args, task)
    # torch_example.knn_experiment()
    torch_example.mean_and_ci_result()
    # d = torch_example.get_cifar10_data()
    # torch_example.divide_to_sub_sets(d)
    # torch_example.mean_and_ci_result()


if __name__ == "__main__":
    # test1()
    main()
