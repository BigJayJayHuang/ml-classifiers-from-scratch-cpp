/*
------------------------------------------------------------
Machine Learning Classifiers in C++
Implements:
 - K-Nearest Neighbors (KNN)
 - Gaussian Naive Bayes
 - Linear Support Vector Machine (SVM)

Author: Jason Huang

Description:
Educational implementation of three common machine learning
algorithms written from scratch using standard C++.

Note:
Implemented from scratch in C++ with assistance from an AI
coding assistant for guidance and debugging.
------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
using namespace std;

//K-Nearest Neighbors classifier
class KNNClassifier {
private:
    int k; //Number of neighnors
    vector<vector<double> > X_train; //Training features
    vector<string> y_train; //Training labels

    //Computes Euclidean distance between two feature vectors
    double euclideanDistance(const vector<double>& point1, const vector<double>& point2) {
        double distance = 0.0;
        for (size_t i = 0; i < point1.size(); ++i) {
            distance += pow(point1[i] - point2[i], 2);
        }
        return sqrt(distance);
    }

    //Returns k nearest neighbors for a given test instance
    vector<pair<double, string> > getNeighbors(const vector<double>& testInstance) {
        vector<pair<double, string> > distances;
        for (size_t i = 0; i < X_train.size(); ++i) {
            double dist = euclideanDistance(testInstance, X_train[i]);
            distances.push_back(make_pair(dist, y_train[i]));
        }
        sort(distances.begin(), distances.end());
        return vector<pair<double, string> >(distances.begin(), distances.begin() + k);
    }

    //Predict label using majority vote
    string predictSingle(const vector<double>& testInstance) {
        vector<pair<double, string> > neighbors = getNeighbors(testInstance);
        unordered_map<string, int> labelCount; //Count frequency of each label among neighbors
        for (const auto& neighbor : neighbors) {
            labelCount[neighbor.second]++;
        }
        return max_element(labelCount.begin(), labelCount.end(), [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }

public:
    KNNClassifier(int kValue) : k(kValue) {}

    //Store training data
    void fit(const vector<vector<double> >& X_trainData, const vector<string>& y_trainData) {
        X_train = X_trainData;
        y_train = y_trainData;
    }

    //Predict labels for many samples
    vector<string> predict(const vector<vector<double> >& X_test) {
        vector<string> predictions;
        for (const auto& instance : X_test) {
            predictions.push_back(predictSingle(instance));
        }
        return predictions;
    }
};

//Gaussian Naive Bayes classifier
class NaiveBayesClassifier {
private:
    unordered_map<string, double> classProbabilities; //P(class)
    unordered_map<string, vector<double> > means; //Feature means
    unordered_map<string, vector<double> > stds; //Feature std dev

    //Gaussian probability density function
    double calculateProbability(double x, double mean, double std) {
        double exponent = exp(-(pow(x - mean, 2) / (2 * pow(std, 2))));
        return (1 / (sqrt(2 * M_PI) * std)) * exponent;
    }

public:
    //Trains classifer using training data
    void fit(const vector<vector<double> >& X_train, const vector<string>& y_train) {
        unordered_map<string, vector<vector<double> > > classData;
        for (size_t i = 0; i < X_train.size(); ++i) {
            classData[y_train[i]].push_back(X_train[i]);
        }

        for (const auto& [label, data] : classData) {
            classProbabilities[label] = static_cast<double>(data.size()) / X_train.size(); //Prior probability P(class)
            int n_features = data[0].size();
            means[label].resize(n_features);
            stds[label].resize(n_features);

            for (size_t i = 0; i < n_features; ++i) {
                vector<double> featureValues;
                for (const auto& instance : data) {
                    featureValues.push_back(instance[i]);
                }
                    double meanValue =
                        accumulate(featureValues.begin(), featureValues.end(), 0.0)
                        / featureValues.size();

                    means[label][i] = meanValue; //Mean of feature values

                    //Standard deviation of feature values
                    stds[label][i] = sqrt(
                        accumulate(featureValues.begin(), featureValues.end(), 0.0,
                            [meanValue](double partialSum, double val) {
                                return partialSum + pow(val - meanValue, 2);
                            }
                        ) / featureValues.size()
                    );
            }
        }
    }

    //Predicts class label for one input sample
    string predictSingle(const vector<double>& inputFeatures) {
        string bestLabel;
        double bestProb = -1.0;

        for (const auto& [label, prob] : classProbabilities) {
            double probClass = prob;
            for (size_t i = 0; i < inputFeatures.size(); ++i) {
                probClass *= calculateProbability(inputFeatures[i], means[label][i], stds[label][i]);
            }
            if (probClass > bestProb) {
                bestProb = probClass;
                bestLabel = label;
            }
        }
        return bestLabel;
    }

    //Predicts class labels for multiple samples
    vector<string> predict(const vector<vector<double> >& X_test) {
        vector<string> predictions;
        for (const auto& instance : X_test) {
            predictions.push_back(predictSingle(instance));
        }
        return predictions;
    }
};

//Linear SVM classifer
class SVMClassifier {
private:
    double learning_rate; //Step size for optimization
    double lambda_param; //REgularization Strength
    int n_iters; //Number of training iterations
    vector<double> w; //Weight vector
    double b; //Bias

    //Dot product between feature vector and weight
    double dotProduct(const vector<double>& x, const vector<double>& wVector) {
        double result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result += x[i] * wVector[i];
        }
        return result;
    }

public:
    SVMClassifier(double lr, double lambdaParam, int iters) : learning_rate(lr), lambda_param(lambdaParam), n_iters(iters) {}

    //Train SVM using gradient descent
    void fit(const vector<vector<double> >& X_trainData, const vector<string>& y_trainData) {
        //Map string labels to +1 / -1 for SVM
        unordered_map<string, int> labelMapping = {{"Female", 1}, {"Male", -1}};
        vector<int> y_trainEncoded;
        for (const auto& label : y_trainData) {
            y_trainEncoded.push_back(labelMapping[label]);
        }

        int n_samples = X_trainData.size();
        int n_features = X_trainData[0].size();
        w = vector<double>(n_features, 0.0);
        b = 0.0;

        for (int iter = 0; iter < n_iters; ++iter) {
            for (size_t idx = 0; idx < X_trainData.size(); ++idx) {
                double condition = y_trainEncoded[idx] * (dotProduct(X_trainData[idx], w) - b);
                if (condition >= 1) {
                    //Only regularization update
                    for (size_t i = 0; i < n_features; ++i) {
                        w[i] -= learning_rate * (2 * lambda_param * w[i]);
                    }
                } else {
                    //Hinge loss + regularization update
                    for (size_t i = 0; i < n_features; ++i) {
                        w[i] -= learning_rate * (2 * lambda_param * w[i] - X_trainData[idx][i] * y_trainEncoded[idx]);
                    }
                    b -= learning_rate * y_trainEncoded[idx];
                }
            }
        }
    }

    //Predict label for one sample
    int predictSingle(const vector<double>& inputFeatures) {
        double approx = dotProduct(inputFeatures, w) - b;
        return (approx >= 0) ? 1 : -1;
    }

    //Predict label for many samples
    vector<int> predict(const vector<vector<double> >& X_test) {
        vector<int> predictions;
        for (const auto& instance : X_test) {
            predictions.push_back(predictSingle(instance));
        }
        return predictions;
    }
};

int main() {
    //Sample data for illustration
    vector<vector<double> > X_train = {{1.2, 3.4}, {2.1, 4.3}, {3.0, 5.2}};
    vector<string> y_train = {"Male", "Female", "Male"};

    vector<vector<double> > X_test = {{1.4, 3.7}, {2.8, 4.1}};

    //Instantiate and train the classifiers
    NaiveBayesClassifier nbClassifier;
    nbClassifier.fit(X_train, y_train);

    KNNClassifier knnClassifier(1);
    knnClassifier.fit(X_train, y_train);

    SVMClassifier svmClassifier(0.001, 0.01, 1000);
    svmClassifier.fit(X_train, y_train);

    //Make predictions using each classifier
    vector<string> nbPredictions = nbClassifier.predict(X_test);
    vector<string> knnPredictions = knnClassifier.predict(X_test);
    vector<int> svmPredictions = svmClassifier.predict(X_test);

    //Output the predictions
    cout << "Naive Bayes Predictions:" << endl;
    for (const auto& prediction : nbPredictions) {
        cout << prediction << endl;
    }
    cout << endl;

    cout << "KNN Predictions:" << endl;
    for (const auto& prediction : knnPredictions) {
        cout << prediction << endl;
    }
    cout << endl;

    cout << "SVM Predictions:" << endl;
    for (const auto& prediction : svmPredictions) {
        cout << (prediction == 1 ? "Male" : "Female") << endl;
    }
    cout << endl;

    return 0;
}
