package models

import (
	"context"
	"github.com/mark3labs/mcp-go/mcp"
)

type Tool struct {
	Definition mcp.Tool
	Handler    func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error)
}

// BigLakeConfiguration represents the BigLakeConfiguration schema from the OpenAPI specification
type BigLakeConfiguration struct {
	Tableformat string `json:"tableFormat,omitempty"` // Required. The table format the metadata only snapshots are stored in.
	Connectionid string `json:"connectionId,omitempty"` // Required. The connection specifying the credentials to be used to read and write to external storage, such as Cloud Storage. The connection_id can have the form "<project\_id>.<location\_id>.<connection\_id>" or "projects/<project\_id>/locations/<location\_id>/connections/<connection\_id>".
	Fileformat string `json:"fileFormat,omitempty"` // Required. The file format the table data is stored in.
	Storageuri string `json:"storageUri,omitempty"` // Required. The fully qualified location prefix of the external folder where table data is stored. The '*' wildcard character is not allowed. The URI should be in the format "gs://bucket/path_to_table/"
}

// JobCreationReason represents the JobCreationReason schema from the OpenAPI specification
type JobCreationReason struct {
	Code string `json:"code,omitempty"` // Output only. Specifies the high level reason why a Job was created.
}

// QueryParameterType represents the QueryParameterType schema from the OpenAPI specification
type QueryParameterType struct {
	Arraytype QueryParameterType `json:"arrayType,omitempty"` // The type of a query parameter.
	Rangeelementtype QueryParameterType `json:"rangeElementType,omitempty"` // The type of a query parameter.
	Structtypes []map[string]interface{} `json:"structTypes,omitempty"` // Optional. The types of the fields of this struct, in order, if this is a struct.
	TypeField string `json:"type,omitempty"` // Required. The top level type of this field.
}

// PrivacyPolicy represents the PrivacyPolicy schema from the OpenAPI specification
type PrivacyPolicy struct {
	Aggregationthresholdpolicy AggregationThresholdPolicy `json:"aggregationThresholdPolicy,omitempty"` // Represents privacy policy associated with "aggregation threshold" method.
}

// HivePartitioningOptions represents the HivePartitioningOptions schema from the OpenAPI specification
type HivePartitioningOptions struct {
	Fields []string `json:"fields,omitempty"` // Output only. For permanent external tables, this field is populated with the hive partition keys in the order they were inferred. The types of the partition keys can be deduced by checking the table schema (which will include the partition keys). Not every API will populate this field in the output. For example, Tables.Get will populate it, but Tables.List will not contain this field.
	Mode string `json:"mode,omitempty"` // Optional. When set, what mode of hive partitioning to use when reading data. The following modes are supported: * AUTO: automatically infer partition key name(s) and type(s). * STRINGS: automatically infer partition key name(s). All types are strings. * CUSTOM: partition key schema is encoded in the source URI prefix. Not all storage formats support hive partitioning. Requesting hive partitioning on an unsupported format will lead to an error. Currently supported formats are: JSON, CSV, ORC, Avro and Parquet.
	Requirepartitionfilter bool `json:"requirePartitionFilter,omitempty"` // Optional. If set to true, queries over this table require a partition filter that can be used for partition elimination to be specified. Note that this field should only be true when creating a permanent external table or querying a temporary external table. Hive-partitioned loads with require_partition_filter explicitly set to true will fail.
	Sourceuriprefix string `json:"sourceUriPrefix,omitempty"` // Optional. When hive partition detection is requested, a common prefix for all source uris must be required. The prefix must end immediately before the partition key encoding begins. For example, consider files following this data layout: gs://bucket/path_to_table/dt=2019-06-01/country=USA/id=7/file.avro gs://bucket/path_to_table/dt=2019-05-31/country=CA/id=3/file.avro When hive partitioning is requested with either AUTO or STRINGS detection, the common prefix can be either of gs://bucket/path_to_table or gs://bucket/path_to_table/. CUSTOM detection requires encoding the partitioning schema immediately after the common prefix. For CUSTOM, any of * gs://bucket/path_to_table/{dt:DATE}/{country:STRING}/{id:INTEGER} * gs://bucket/path_to_table/{dt:STRING}/{country:STRING}/{id:INTEGER} * gs://bucket/path_to_table/{dt:DATE}/{country:STRING}/{id:STRING} would all be valid source URI prefixes.
}

// TableConstraints represents the TableConstraints schema from the OpenAPI specification
type TableConstraints struct {
	Foreignkeys []map[string]interface{} `json:"foreignKeys,omitempty"` // Optional. Present only if the table has a foreign key. The foreign key is not enforced.
	Primarykey map[string]interface{} `json:"primaryKey,omitempty"` // Represents the primary key constraint on a table's columns.
}

// TableReference represents the TableReference schema from the OpenAPI specification
type TableReference struct {
	Datasetid string `json:"datasetId,omitempty"` // Required. The ID of the dataset containing this table.
	Projectid string `json:"projectId,omitempty"` // Required. The ID of the project containing this table.
	Tableid string `json:"tableId,omitempty"` // Required. The ID of the table. The ID can contain Unicode characters in category L (letter), M (mark), N (number), Pc (connector, including underscore), Pd (dash), and Zs (space). For more information, see [General Category](https://wikipedia.org/wiki/Unicode_character_property#General_Category). The maximum length is 1,024 characters. Certain operations allow suffixing of the table ID with a partition decorator, such as `sample_table$20190123`.
}

// BinaryClassificationMetrics represents the BinaryClassificationMetrics schema from the OpenAPI specification
type BinaryClassificationMetrics struct {
	Binaryconfusionmatrixlist []BinaryConfusionMatrix `json:"binaryConfusionMatrixList,omitempty"` // Binary confusion matrix at multiple thresholds.
	Negativelabel string `json:"negativeLabel,omitempty"` // Label representing the negative class.
	Positivelabel string `json:"positiveLabel,omitempty"` // Label representing the positive class.
	Aggregateclassificationmetrics AggregateClassificationMetrics `json:"aggregateClassificationMetrics,omitempty"` // Aggregate metrics for classification/classifier models. For multi-class models, the metrics are either macro-averaged or micro-averaged. When macro-averaged, the metrics are calculated for each label and then an unweighted average is taken of those values. When micro-averaged, the metric is calculated globally by counting the total number of correctly predicted rows.
}

// RankingMetrics represents the RankingMetrics schema from the OpenAPI specification
type RankingMetrics struct {
	Meansquarederror float64 `json:"meanSquaredError,omitempty"` // Similar to the mean squared error computed in regression and explicit recommendation models except instead of computing the rating directly, the output from evaluate is computed against a preference which is 1 or 0 depending on if the rating exists or not.
	Normalizeddiscountedcumulativegain float64 `json:"normalizedDiscountedCumulativeGain,omitempty"` // A metric to determine the goodness of a ranking calculated from the predicted confidence by comparing it to an ideal rank measured by the original ratings.
	Averagerank float64 `json:"averageRank,omitempty"` // Determines the goodness of a ranking by computing the percentile rank from the predicted confidence and dividing it by the original rank.
	Meanaverageprecision float64 `json:"meanAveragePrecision,omitempty"` // Calculates a precision per user for all the items by ranking them and then averages all the precisions across all the users.
}

// RegressionMetrics represents the RegressionMetrics schema from the OpenAPI specification
type RegressionMetrics struct {
	Meanabsoluteerror float64 `json:"meanAbsoluteError,omitempty"` // Mean absolute error.
	Meansquarederror float64 `json:"meanSquaredError,omitempty"` // Mean squared error.
	Meansquaredlogerror float64 `json:"meanSquaredLogError,omitempty"` // Mean squared log error.
	Medianabsoluteerror float64 `json:"medianAbsoluteError,omitempty"` // Median absolute error.
	Rsquared float64 `json:"rSquared,omitempty"` // R^2 score. This corresponds to r2_score in ML.EVALUATE.
}

// JobConfigurationTableCopy represents the JobConfigurationTableCopy schema from the OpenAPI specification
type JobConfigurationTableCopy struct {
	Destinationtable TableReference `json:"destinationTable,omitempty"`
	Operationtype string `json:"operationType,omitempty"` // Optional. Supported operation types in table copy job.
	Sourcetable TableReference `json:"sourceTable,omitempty"`
	Sourcetables []TableReference `json:"sourceTables,omitempty"` // [Pick one] Source tables to copy.
	Writedisposition string `json:"writeDisposition,omitempty"` // Optional. Specifies the action that occurs if the destination table already exists. The following values are supported: * WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the table data and uses the schema and table constraints from the source table. * WRITE_APPEND: If the table already exists, BigQuery appends the data to the table. * WRITE_EMPTY: If the table already exists and contains data, a 'duplicate' error is returned in the job result. The default value is WRITE_EMPTY. Each action is atomic and only occurs if BigQuery is able to complete the job successfully. Creation, truncation and append actions occur as one atomic update upon job completion.
	Createdisposition string `json:"createDisposition,omitempty"` // Optional. Specifies whether the job is allowed to create new tables. The following values are supported: * CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the table. * CREATE_NEVER: The table must already exist. If it does not, a 'notFound' error is returned in the job result. The default value is CREATE_IF_NEEDED. Creation, truncation and append actions occur as one atomic update upon job completion.
	Destinationencryptionconfiguration EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`
	Destinationexpirationtime string `json:"destinationExpirationTime,omitempty"` // Optional. The time when the destination table expires. Expired tables will be deleted and their storage reclaimed.
}

// QueryParameter represents the QueryParameter schema from the OpenAPI specification
type QueryParameter struct {
	Parametertype QueryParameterType `json:"parameterType,omitempty"` // The type of a query parameter.
	Parametervalue QueryParameterValue `json:"parameterValue,omitempty"` // The value of a query parameter.
	Name string `json:"name,omitempty"` // Optional. If unset, this is a positional parameter. Otherwise, should be unique within a query.
}

// TrainingOptions represents the TrainingOptions schema from the OpenAPI specification
type TrainingOptions struct {
	Vertexaimodelversionaliases []string `json:"vertexAiModelVersionAliases,omitempty"` // The version aliases to apply in Vertex AI model registry. Always overwrite if the version aliases exists in a existing model.
	Datasplitmethod string `json:"dataSplitMethod,omitempty"` // The data split type for training and evaluation, e.g. RANDOM.
	Pcasolver string `json:"pcaSolver,omitempty"` // The solver for PCA.
	Colsamplebylevel float64 `json:"colsampleBylevel,omitempty"` // Subsample ratio of columns for each level for boosted tree models.
	Hparamtuningobjectives []string `json:"hparamTuningObjectives,omitempty"` // The target evaluation metrics to optimize the hyperparameters for.
	Xgboostversion string `json:"xgboostVersion,omitempty"` // User-selected XGBoost versions for training of XGBoost models.
	Losstype string `json:"lossType,omitempty"` // Type of loss function used during training run.
	Batchsize string `json:"batchSize,omitempty"` // Batch size for dnn models.
	Learnratestrategy string `json:"learnRateStrategy,omitempty"` // The strategy to determine learn rate for the current iteration.
	Modeluri string `json:"modelUri,omitempty"` // Google Cloud Storage URI from which the model was imported. Only applicable for imported models.
	Maxparalleltrials string `json:"maxParallelTrials,omitempty"` // Maximum number of trials to run in parallel.
	Sampledshapleynumpaths string `json:"sampledShapleyNumPaths,omitempty"` // Number of paths for the sampled Shapley explain method.
	Trendsmoothingwindowsize string `json:"trendSmoothingWindowSize,omitempty"` // Smoothing window size for the trend component. When a positive value is specified, a center moving average smoothing is applied on the history trend. When the smoothing window is out of the boundary at the beginning or the end of the trend, the first element or the last element is padded to fill the smoothing window before the average is applied.
	Calculatepvalues bool `json:"calculatePValues,omitempty"` // Whether or not p-value test should be computed for this model. Only available for linear and logistic regression models.
	Kmeansinitializationmethod string `json:"kmeansInitializationMethod,omitempty"` // The method used to initialize the centroids for kmeans algorithm.
	Enableglobalexplain bool `json:"enableGlobalExplain,omitempty"` // If true, enable global explanation during training.
	Treemethod string `json:"treeMethod,omitempty"` // Tree construction algorithm for boosted tree models.
	Optimizationstrategy string `json:"optimizationStrategy,omitempty"` // Optimization strategy for training linear regression models.
	Fitintercept bool `json:"fitIntercept,omitempty"` // Whether the model should include intercept during model training.
	Autoarimamaxorder string `json:"autoArimaMaxOrder,omitempty"` // The max value of the sum of non-seasonal p and q.
	Integratedgradientsnumsteps string `json:"integratedGradientsNumSteps,omitempty"` // Number of integral steps for the integrated gradients explain method.
	Autoarimaminorder string `json:"autoArimaMinOrder,omitempty"` // The min value of the sum of non-seasonal p and q.
	Instanceweightcolumn string `json:"instanceWeightColumn,omitempty"` // Name of the instance weight column for training data. This column isn't be used as a feature.
	L2regularization float64 `json:"l2Regularization,omitempty"` // L2 regularization coefficient.
	Inputlabelcolumns []string `json:"inputLabelColumns,omitempty"` // Name of input label columns in training data.
	Dartnormalizetype string `json:"dartNormalizeType,omitempty"` // Type of normalization algorithm for boosted tree models using dart booster.
	Autoclassweights bool `json:"autoClassWeights,omitempty"` // Whether to calculate class weights automatically based on the popularity of each label.
	L1regactivation float64 `json:"l1RegActivation,omitempty"` // L1 regularization coefficient to activations.
	Timeseriesidcolumns []string `json:"timeSeriesIdColumns,omitempty"` // The time series id columns that were used during ARIMA model training.
	Holidayregions []string `json:"holidayRegions,omitempty"` // A list of geographical regions that are used for time series modeling.
	Decomposetimeseries bool `json:"decomposeTimeSeries,omitempty"` // If true, perform decompose time series and save the results.
	Datasplitevalfraction float64 `json:"dataSplitEvalFraction,omitempty"` // The fraction of evaluation data over the whole input data. The rest of data will be used as training data. The format should be double. Accurate to two decimal places. Default value is 0.2.
	Earlystop bool `json:"earlyStop,omitempty"` // Whether to stop early when the loss doesn't improve significantly any more (compared to min_relative_progress). Used only for iterative training algorithms.
	Walsalpha float64 `json:"walsAlpha,omitempty"` // Hyperparameter for matrix factoration when implicit feedback type is specified.
	Maxtreedepth string `json:"maxTreeDepth,omitempty"` // Maximum depth of a tree for boosted tree models.
	Approxglobalfeaturecontrib bool `json:"approxGlobalFeatureContrib,omitempty"` // Whether to use approximate feature contribution method in XGBoost model explanation for global explain.
	Tfversion string `json:"tfVersion,omitempty"` // Based on the selected TF version, the corresponding docker image is used to train external models.
	Maxtimeserieslength string `json:"maxTimeSeriesLength,omitempty"` // The maximum number of time points in a time series that can be used in modeling the trend component of the time series. Don't use this option with the `timeSeriesLengthFraction` or `minTimeSeriesLength` options.
	Numclusters string `json:"numClusters,omitempty"` // Number of clusters for clustering models.
	Standardizefeatures bool `json:"standardizeFeatures,omitempty"` // Whether to standardize numerical features. Default to true.
	Warmstart bool `json:"warmStart,omitempty"` // Whether to train a model from the last checkpoint.
	Minrelativeprogress float64 `json:"minRelativeProgress,omitempty"` // When early_stop is true, stops training when accuracy improvement is less than 'min_relative_progress'. Used only for iterative training algorithms.
	Timeseriestimestampcolumn string `json:"timeSeriesTimestampColumn,omitempty"` // Column to be designated as time series timestamp for ARIMA model.
	Horizon string `json:"horizon,omitempty"` // The number of periods ahead that need to be forecasted.
	Datafrequency string `json:"dataFrequency,omitempty"` // The data frequency of a time series.
	Nonseasonalorder ArimaOrder `json:"nonSeasonalOrder,omitempty"` // Arima order, can be used for both non-seasonal and seasonal parts.
	Budgethours float64 `json:"budgetHours,omitempty"` // Budget in hours for AutoML training.
	Kmeansinitializationcolumn string `json:"kmeansInitializationColumn,omitempty"` // The column used to provide the initial centroids for kmeans algorithm when kmeans_initialization_method is CUSTOM.
	Colsamplebynode float64 `json:"colsampleBynode,omitempty"` // Subsample ratio of columns for each node(split) for boosted tree models.
	Numprincipalcomponents string `json:"numPrincipalComponents,omitempty"` // Number of principal components to keep in the PCA model. Must be <= the number of features.
	Hiddenunits []string `json:"hiddenUnits,omitempty"` // Hidden units for dnn models.
	Mintimeserieslength string `json:"minTimeSeriesLength,omitempty"` // The minimum number of time points in a time series that are used in modeling the trend component of the time series. If you use this option you must also set the `timeSeriesLengthFraction` option. This training option ensures that enough time points are available when you use `timeSeriesLengthFraction` in trend modeling. This is particularly important when forecasting multiple time series in a single query using `timeSeriesIdColumn`. If the total number of time points is less than the `minTimeSeriesLength` value, then the query uses all available time points.
	Modelregistry string `json:"modelRegistry,omitempty"` // The model registry.
	Timeseriesidcolumn string `json:"timeSeriesIdColumn,omitempty"` // The time series id column that was used during ARIMA model training.
	Optimizer string `json:"optimizer,omitempty"` // Optimizer used for training the neural nets.
	Categoryencodingmethod string `json:"categoryEncodingMethod,omitempty"` // Categorical feature encoding method.
	Cleanspikesanddips bool `json:"cleanSpikesAndDips,omitempty"` // If true, clean spikes and dips in the input time series.
	Usercolumn string `json:"userColumn,omitempty"` // User column specified for matrix factorization models.
	Includedrift bool `json:"includeDrift,omitempty"` // Include drift when fitting an ARIMA model.
	Pcaexplainedvarianceratio float64 `json:"pcaExplainedVarianceRatio,omitempty"` // The minimum ratio of cumulative explained variance that needs to be given by the PCA model.
	Holidayregion string `json:"holidayRegion,omitempty"` // The geographical region based on which the holidays are considered in time series modeling. If a valid value is specified, then holiday effects modeling is enabled.
	Learnrate float64 `json:"learnRate,omitempty"` // Learning rate in training. Used only for iterative training algorithms.
	Labelclassweights map[string]interface{} `json:"labelClassWeights,omitempty"` // Weights associated with each label class, for rebalancing the training data. Only applicable for classification models.
	Colsamplebytree float64 `json:"colsampleBytree,omitempty"` // Subsample ratio of columns when constructing each tree for boosted tree models.
	Autoarima bool `json:"autoArima,omitempty"` // Whether to enable auto ARIMA or not.
	Distancetype string `json:"distanceType,omitempty"` // Distance type for clustering models.
	Adjuststepchanges bool `json:"adjustStepChanges,omitempty"` // If true, detect step changes and make data adjustment in the input time series.
	Datasplitcolumn string `json:"dataSplitColumn,omitempty"` // The column to split data with. This column won't be used as a feature. 1. When data_split_method is CUSTOM, the corresponding column should be boolean. The rows with true value tag are eval data, and the false are training data. 2. When data_split_method is SEQ, the first DATA_SPLIT_EVAL_FRACTION rows (from smallest to largest) in the corresponding column are used as training data, and the rest are eval data. It respects the order in Orderable data types: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#data-type-properties
	Mintreechildweight string `json:"minTreeChildWeight,omitempty"` // Minimum sum of instance weight needed in a child for boosted tree models.
	Initiallearnrate float64 `json:"initialLearnRate,omitempty"` // Specifies the initial learning rate for the line search learn rate strategy.
	Activationfn string `json:"activationFn,omitempty"` // Activation function of the neural nets.
	Itemcolumn string `json:"itemColumn,omitempty"` // Item column specified for matrix factorization models.
	L1regularization float64 `json:"l1Regularization,omitempty"` // L1 regularization coefficient.
	Boostertype string `json:"boosterType,omitempty"` // Booster type for boosted tree models.
	Minsplitloss float64 `json:"minSplitLoss,omitempty"` // Minimum split loss for boosted tree models.
	Numtrials string `json:"numTrials,omitempty"` // Number of trials to run this hyperparameter tuning job.
	Colorspace string `json:"colorSpace,omitempty"` // Enums for color space, used for processing images in Object Table. See more details at https://www.tensorflow.org/io/tutorials/colorspace.
	Feedbacktype string `json:"feedbackType,omitempty"` // Feedback type that specifies which algorithm to run for matrix factorization.
	Timeseriesdatacolumn string `json:"timeSeriesDataColumn,omitempty"` // Column to be designated as time series data for ARIMA model.
	Subsample float64 `json:"subsample,omitempty"` // Subsample fraction of the training data to grow tree to prevent overfitting for boosted tree models.
	Timeserieslengthfraction float64 `json:"timeSeriesLengthFraction,omitempty"` // The fraction of the interpolated length of the time series that's used to model the time series trend component. All of the time points of the time series are used to model the non-trend component. This training option accelerates modeling training without sacrificing much forecasting accuracy. You can use this option with `minTimeSeriesLength` but not with `maxTimeSeriesLength`.
	Dropout float64 `json:"dropout,omitempty"` // Dropout probability for dnn models.
	Maxiterations string `json:"maxIterations,omitempty"` // The maximum number of iterations in training. Used only for iterative training algorithms.
	Numfactors string `json:"numFactors,omitempty"` // Num factors specified for matrix factorization models.
	Numparalleltree string `json:"numParallelTree,omitempty"` // Number of parallel trees constructed during each iteration for boosted tree models.
	Scalefeatures bool `json:"scaleFeatures,omitempty"` // If true, scale the feature values by dividing the feature standard deviation. Currently only apply to PCA.
}

// TableDataInsertAllResponse represents the TableDataInsertAllResponse schema from the OpenAPI specification
type TableDataInsertAllResponse struct {
	Inserterrors []map[string]interface{} `json:"insertErrors,omitempty"` // Describes specific errors encountered while processing the request.
	Kind string `json:"kind,omitempty"` // Returns "bigquery#tableDataInsertAllResponse".
}

// MaterializedViewDefinition represents the MaterializedViewDefinition schema from the OpenAPI specification
type MaterializedViewDefinition struct {
	Maxstaleness string `json:"maxStaleness,omitempty"` // [Optional] Max staleness of data that could be returned when materizlized view is queried (formatted as Google SQL Interval type).
	Query string `json:"query,omitempty"` // Required. A query whose results are persisted.
	Refreshintervalms string `json:"refreshIntervalMs,omitempty"` // Optional. The maximum frequency at which this materialized view will be refreshed. The default value is "1800000" (30 minutes).
	Allownonincrementaldefinition bool `json:"allowNonIncrementalDefinition,omitempty"` // Optional. This option declares authors intention to construct a materialized view that will not be refreshed incrementally.
	Enablerefresh bool `json:"enableRefresh,omitempty"` // Optional. Enable automatic refresh of the materialized view when the base table is updated. The default value is "true".
	Lastrefreshtime string `json:"lastRefreshTime,omitempty"` // Output only. The time when this materialized view was last refreshed, in milliseconds since the epoch.
}

// DataFormatOptions represents the DataFormatOptions schema from the OpenAPI specification
type DataFormatOptions struct {
	Useint64timestamp bool `json:"useInt64Timestamp,omitempty"` // Optional. Output timestamp as usec int64. Default is false.
}

// ProjectReference represents the ProjectReference schema from the OpenAPI specification
type ProjectReference struct {
	Projectid string `json:"projectId,omitempty"` // Required. ID of the project. Can be either the numeric ID or the assigned ID of the project.
}

// FeatureValue represents the FeatureValue schema from the OpenAPI specification
type FeatureValue struct {
	Categoricalvalue CategoricalValue `json:"categoricalValue,omitempty"` // Representative value of a categorical feature.
	Featurecolumn string `json:"featureColumn,omitempty"` // The feature column name.
	Numericalvalue float64 `json:"numericalValue,omitempty"` // The numerical feature value. This is the centroid value for this feature.
}

// ListRoutinesResponse represents the ListRoutinesResponse schema from the OpenAPI specification
type ListRoutinesResponse struct {
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token to request the next page of results.
	Routines []Routine `json:"routines,omitempty"` // Routines in the requested dataset. Unless read_mask is set in the request, only the following fields are populated: etag, project_id, dataset_id, routine_id, routine_type, creation_time, last_modified_time, language, and remote_function_options.
}

// ConfusionMatrix represents the ConfusionMatrix schema from the OpenAPI specification
type ConfusionMatrix struct {
	Rows []Row `json:"rows,omitempty"` // One row per actual label.
	Confidencethreshold float64 `json:"confidenceThreshold,omitempty"` // Confidence threshold used when computing the entries of the confusion matrix.
}

// AuditConfig represents the AuditConfig schema from the OpenAPI specification
type AuditConfig struct {
	Auditlogconfigs []AuditLogConfig `json:"auditLogConfigs,omitempty"` // The configuration for logging of each type of permission.
	Service string `json:"service,omitempty"` // Specifies a service that will be enabled for audit logging. For example, `storage.googleapis.com`, `cloudsql.googleapis.com`. `allServices` is a special value that covers all services.
}

// SessionInfo represents the SessionInfo schema from the OpenAPI specification
type SessionInfo struct {
	Sessionid string `json:"sessionId,omitempty"` // Output only. The id of the session.
}

// JobConfigurationQuery represents the JobConfigurationQuery schema from the OpenAPI specification
type JobConfigurationQuery struct {
	Destinationtable TableReference `json:"destinationTable,omitempty"`
	Scriptoptions ScriptOptions `json:"scriptOptions,omitempty"` // Options related to script execution.
	Timepartitioning TimePartitioning `json:"timePartitioning,omitempty"`
	Clustering Clustering `json:"clustering,omitempty"` // Configures table clustering.
	Uselegacysql bool `json:"useLegacySql,omitempty"` // Optional. Specifies whether to use BigQuery's legacy SQL dialect for this query. The default value is true. If set to false, the query will use BigQuery's GoogleSQL: https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is set to false, the value of flattenResults is ignored; query will be run as if flattenResults is false.
	Usequerycache bool `json:"useQueryCache,omitempty"` // Optional. Whether to look for the result in the query cache. The query cache is a best-effort cache that will be flushed whenever tables in the query are modified. Moreover, the query cache is only available when a query does not have a destination table specified. The default value is true.
	Writedisposition string `json:"writeDisposition,omitempty"` // Optional. Specifies the action that occurs if the destination table already exists. The following values are supported: * WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the data, removes the constraints, and uses the schema from the query result. * WRITE_APPEND: If the table already exists, BigQuery appends the data to the table. * WRITE_EMPTY: If the table already exists and contains data, a 'duplicate' error is returned in the job result. The default value is WRITE_EMPTY. Each action is atomic and only occurs if BigQuery is able to complete the job successfully. Creation, truncation and append actions occur as one atomic update upon job completion.
	Schemaupdateoptions []string `json:"schemaUpdateOptions,omitempty"` // Allows the schema of the destination table to be updated as a side effect of the query job. Schema update options are supported in two cases: when writeDisposition is WRITE_APPEND; when writeDisposition is WRITE_TRUNCATE and the destination table is a partition of a table, specified by partition decorators. For normal tables, WRITE_TRUNCATE will always overwrite the schema. One or more of the following values are specified: * ALLOW_FIELD_ADDITION: allow adding a nullable field to the schema. * ALLOW_FIELD_RELAXATION: allow relaxing a required field in the original schema to nullable.
	Createsession bool `json:"createSession,omitempty"` // If this property is true, the job creates a new session using a randomly generated session_id. To continue using a created session with subsequent queries, pass the existing session identifier as a `ConnectionProperty` value. The session identifier is returned as part of the `SessionInfo` message within the query statistics. The new session's location will be set to `Job.JobReference.location` if it is present, otherwise it's set to the default location based on existing routing logic.
	Query string `json:"query,omitempty"` // [Required] SQL query text to execute. The useLegacySql field can be used to indicate whether the query uses legacy SQL or GoogleSQL.
	Systemvariables SystemVariables `json:"systemVariables,omitempty"` // System variables given to a query.
	Priority string `json:"priority,omitempty"` // Optional. Specifies a priority for the query. Possible values include INTERACTIVE and BATCH. The default value is INTERACTIVE.
	Userdefinedfunctionresources []UserDefinedFunctionResource `json:"userDefinedFunctionResources,omitempty"` // Describes user-defined function resources used in the query.
	Allowlargeresults bool `json:"allowLargeResults,omitempty"` // Optional. If true and query uses legacy SQL dialect, allows the query to produce arbitrarily large result tables at a slight cost in performance. Requires destinationTable to be set. For GoogleSQL queries, this flag is ignored and large results are always allowed. However, you must still set destinationTable when result size exceeds the allowed maximum response size.
	Continuous bool `json:"continuous,omitempty"` // [Optional] Specifies whether the query should be executed as a continuous query. The default value is false.
	Maximumbillingtier int `json:"maximumBillingTier,omitempty"` // Optional. [Deprecated] Maximum billing tier allowed for this query. The billing tier controls the amount of compute resources allotted to the query, and multiplies the on-demand cost of the query accordingly. A query that runs within its allotted resources will succeed and indicate its billing tier in statistics.query.billingTier, but if the query exceeds its allotted resources, it will fail with billingTierLimitExceeded. WARNING: The billed byte amount can be multiplied by an amount up to this number! Most users should not need to alter this setting, and we recommend that you avoid introducing new uses of it.
	Createdisposition string `json:"createDisposition,omitempty"` // Optional. Specifies whether the job is allowed to create new tables. The following values are supported: * CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the table. * CREATE_NEVER: The table must already exist. If it does not, a 'notFound' error is returned in the job result. The default value is CREATE_IF_NEEDED. Creation, truncation and append actions occur as one atomic update upon job completion.
	Maximumbytesbilled string `json:"maximumBytesBilled,omitempty"` // Limits the bytes billed for this job. Queries that will have bytes billed beyond this limit will fail (without incurring a charge). If unspecified, this will be set to your project default.
	Parametermode string `json:"parameterMode,omitempty"` // GoogleSQL only. Set to POSITIONAL to use positional (?) query parameters or to NAMED to use named (@myparam) query parameters in this query.
	Queryparameters []QueryParameter `json:"queryParameters,omitempty"` // Query parameters for GoogleSQL queries.
	Preservenulls bool `json:"preserveNulls,omitempty"` // [Deprecated] This property is deprecated.
	Rangepartitioning RangePartitioning `json:"rangePartitioning,omitempty"`
	Tabledefinitions map[string]interface{} `json:"tableDefinitions,omitempty"` // Optional. You can specify external table definitions, which operate as ephemeral tables that can be queried. These definitions are configured using a JSON map, where the string key represents the table identifier, and the value is the corresponding external data configuration object.
	Connectionproperties []ConnectionProperty `json:"connectionProperties,omitempty"` // Connection properties which can modify the query behavior.
	Destinationencryptionconfiguration EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`
	Flattenresults bool `json:"flattenResults,omitempty"` // Optional. If true and query uses legacy SQL dialect, flattens all nested and repeated fields in the query results. allowLargeResults must be true if this is set to false. For GoogleSQL queries, this flag is ignored and results are never flattened.
	Defaultdataset DatasetReference `json:"defaultDataset,omitempty"`
}

// ClusterInfo represents the ClusterInfo schema from the OpenAPI specification
type ClusterInfo struct {
	Clustersize string `json:"clusterSize,omitempty"` // Cluster size, the total number of points assigned to the cluster.
	Centroidid string `json:"centroidId,omitempty"` // Centroid id.
	Clusterradius float64 `json:"clusterRadius,omitempty"` // Cluster radius, the average distance from centroid to each point assigned to the cluster.
}

// JobStatistics2 represents the JobStatistics2 schema from the OpenAPI specification
type JobStatistics2 struct {
	Timeline []QueryTimelineSample `json:"timeline,omitempty"` // Output only. Describes a timeline of job execution.
	Numdmlaffectedrows string `json:"numDmlAffectedRows,omitempty"` // Output only. The number of rows affected by a DML statement. Present only for DML statements INSERT, UPDATE or DELETE.
	Materializedviewstatistics MaterializedViewStatistics `json:"materializedViewStatistics,omitempty"` // Statistics of materialized views considered in a query job.
	Ddltargetdataset DatasetReference `json:"ddlTargetDataset,omitempty"`
	Cachehit bool `json:"cacheHit,omitempty"` // Output only. Whether the query result was fetched from the query cache.
	Searchstatistics SearchStatistics `json:"searchStatistics,omitempty"` // Statistics for a search query. Populated as part of JobStatistics2.
	Vectorsearchstatistics VectorSearchStatistics `json:"vectorSearchStatistics,omitempty"` // Statistics for a vector search query. Populated as part of JobStatistics2.
	Exportdatastatistics ExportDataStatistics `json:"exportDataStatistics,omitempty"` // Statistics for the EXPORT DATA statement as part of Query Job. EXTRACT JOB statistics are populated in JobStatistics4.
	Totalslotms string `json:"totalSlotMs,omitempty"` // Output only. Slot-milliseconds for the job.
	Dcltargetdataset DatasetReference `json:"dclTargetDataset,omitempty"`
	Ddltargettable TableReference `json:"ddlTargetTable,omitempty"`
	Mlstatistics MlStatistics `json:"mlStatistics,omitempty"` // Job statistics specific to a BigQuery ML training job.
	Reservationusage []map[string]interface{} `json:"reservationUsage,omitempty"` // Output only. Job resource usage breakdown by reservation. This field reported misleading information and will no longer be populated.
	Metadatacachestatistics MetadataCacheStatistics `json:"metadataCacheStatistics,omitempty"` // Statistics for metadata caching in BigLake tables.
	Referencedroutines []RoutineReference `json:"referencedRoutines,omitempty"` // Output only. Referenced routines for the job.
	Totalbytesprocessed string `json:"totalBytesProcessed,omitempty"` // Output only. Total bytes processed for the job.
	Transferredbytes string `json:"transferredBytes,omitempty"` // Output only. Total bytes transferred for cross-cloud queries such as Cross Cloud Transfer and CREATE TABLE AS SELECT (CTAS).
	Modeltrainingcurrentiteration int `json:"modelTrainingCurrentIteration,omitempty"` // Deprecated.
	Loadquerystatistics LoadQueryStatistics `json:"loadQueryStatistics,omitempty"` // Statistics for a LOAD query.
	Estimatedbytesprocessed string `json:"estimatedBytesProcessed,omitempty"` // Output only. The original estimate of bytes processed for the job.
	Billingtier int `json:"billingTier,omitempty"` // Output only. Billing tier for the job. This is a BigQuery-specific concept which is not related to the Google Cloud notion of "free tier". The value here is a measure of the query's resource consumption relative to the amount of data scanned. For on-demand queries, the limit is 100, and all queries within this limit are billed at the standard on-demand rates. On-demand queries that exceed this limit will fail with a billingTierLimitExceeded error.
	Ddloperationperformed string `json:"ddlOperationPerformed,omitempty"` // Output only. The DDL operation performed, possibly dependent on the pre-existence of the DDL target.
	Modeltraining BigQueryModelTraining `json:"modelTraining,omitempty"`
	Queryinfo QueryInfo `json:"queryInfo,omitempty"` // Query optimization information for a QUERY job.
	Bienginestatistics BiEngineStatistics `json:"biEngineStatistics,omitempty"` // Statistics for a BI Engine specific query. Populated as part of JobStatistics2
	Sparkstatistics SparkStatistics `json:"sparkStatistics,omitempty"` // Statistics for a BigSpark query. Populated as part of JobStatistics2
	Dcltargettable TableReference `json:"dclTargetTable,omitempty"`
	Ddltargetroutine RoutineReference `json:"ddlTargetRoutine,omitempty"` // Id path of a routine.
	Modeltrainingexpectedtotaliteration string `json:"modelTrainingExpectedTotalIteration,omitempty"` // Deprecated.
	Referencedtables []TableReference `json:"referencedTables,omitempty"` // Output only. Referenced tables for the job. Queries that reference more than 50 tables will not have a complete list.
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
	Ddltargetrowaccesspolicy RowAccessPolicyReference `json:"ddlTargetRowAccessPolicy,omitempty"` // Id path of a row access policy.
	Ddldestinationtable TableReference `json:"ddlDestinationTable,omitempty"`
	Totalpartitionsprocessed string `json:"totalPartitionsProcessed,omitempty"` // Output only. Total number of partitions processed from all partitioned tables referenced in the job.
	Dmlstats DmlStatistics `json:"dmlStats,omitempty"` // Detailed statistics for DML statements
	Statementtype string `json:"statementType,omitempty"` // Output only. The type of query statement, if valid. Possible values: * `SELECT`: [`SELECT`](/bigquery/docs/reference/standard-sql/query-syntax#select_list) statement. * `ASSERT`: [`ASSERT`](/bigquery/docs/reference/standard-sql/debugging-statements#assert) statement. * `INSERT`: [`INSERT`](/bigquery/docs/reference/standard-sql/dml-syntax#insert_statement) statement. * `UPDATE`: [`UPDATE`](/bigquery/docs/reference/standard-sql/query-syntax#update_statement) statement. * `DELETE`: [`DELETE`](/bigquery/docs/reference/standard-sql/data-manipulation-language) statement. * `MERGE`: [`MERGE`](/bigquery/docs/reference/standard-sql/data-manipulation-language) statement. * `CREATE_TABLE`: [`CREATE TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#create_table_statement) statement, without `AS SELECT`. * `CREATE_TABLE_AS_SELECT`: [`CREATE TABLE AS SELECT`](/bigquery/docs/reference/standard-sql/data-definition-language#query_statement) statement. * `CREATE_VIEW`: [`CREATE VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#create_view_statement) statement. * `CREATE_MODEL`: [`CREATE MODEL`](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create#create_model_statement) statement. * `CREATE_MATERIALIZED_VIEW`: [`CREATE MATERIALIZED VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#create_materialized_view_statement) statement. * `CREATE_FUNCTION`: [`CREATE FUNCTION`](/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement) statement. * `CREATE_TABLE_FUNCTION`: [`CREATE TABLE FUNCTION`](/bigquery/docs/reference/standard-sql/data-definition-language#create_table_function_statement) statement. * `CREATE_PROCEDURE`: [`CREATE PROCEDURE`](/bigquery/docs/reference/standard-sql/data-definition-language#create_procedure) statement. * `CREATE_ROW_ACCESS_POLICY`: [`CREATE ROW ACCESS POLICY`](/bigquery/docs/reference/standard-sql/data-definition-language#create_row_access_policy_statement) statement. * `CREATE_SCHEMA`: [`CREATE SCHEMA`](/bigquery/docs/reference/standard-sql/data-definition-language#create_schema_statement) statement. * `CREATE_SNAPSHOT_TABLE`: [`CREATE SNAPSHOT TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#create_snapshot_table_statement) statement. * `CREATE_SEARCH_INDEX`: [`CREATE SEARCH INDEX`](/bigquery/docs/reference/standard-sql/data-definition-language#create_search_index_statement) statement. * `DROP_TABLE`: [`DROP TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_table_statement) statement. * `DROP_EXTERNAL_TABLE`: [`DROP EXTERNAL TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_external_table_statement) statement. * `DROP_VIEW`: [`DROP VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_view_statement) statement. * `DROP_MODEL`: [`DROP MODEL`](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-drop-model) statement. * `DROP_MATERIALIZED_VIEW`: [`DROP MATERIALIZED VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_materialized_view_statement) statement. * `DROP_FUNCTION` : [`DROP FUNCTION`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_function_statement) statement. * `DROP_TABLE_FUNCTION` : [`DROP TABLE FUNCTION`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_table_function) statement. * `DROP_PROCEDURE`: [`DROP PROCEDURE`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_procedure_statement) statement. * `DROP_SEARCH_INDEX`: [`DROP SEARCH INDEX`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_search_index) statement. * `DROP_SCHEMA`: [`DROP SCHEMA`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_schema_statement) statement. * `DROP_SNAPSHOT_TABLE`: [`DROP SNAPSHOT TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_snapshot_table_statement) statement. * `DROP_ROW_ACCESS_POLICY`: [`DROP [ALL] ROW ACCESS POLICY|POLICIES`](/bigquery/docs/reference/standard-sql/data-definition-language#drop_row_access_policy_statement) statement. * `ALTER_TABLE`: [`ALTER TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#alter_table_set_options_statement) statement. * `ALTER_VIEW`: [`ALTER VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#alter_view_set_options_statement) statement. * `ALTER_MATERIALIZED_VIEW`: [`ALTER MATERIALIZED VIEW`](/bigquery/docs/reference/standard-sql/data-definition-language#alter_materialized_view_set_options_statement) statement. * `ALTER_SCHEMA`: [`ALTER SCHEMA`](/bigquery/docs/reference/standard-sql/data-definition-language#aalter_schema_set_options_statement) statement. * `SCRIPT`: [`SCRIPT`](/bigquery/docs/reference/standard-sql/procedural-language). * `TRUNCATE_TABLE`: [`TRUNCATE TABLE`](/bigquery/docs/reference/standard-sql/dml-syntax#truncate_table_statement) statement. * `CREATE_EXTERNAL_TABLE`: [`CREATE EXTERNAL TABLE`](/bigquery/docs/reference/standard-sql/data-definition-language#create_external_table_statement) statement. * `EXPORT_DATA`: [`EXPORT DATA`](/bigquery/docs/reference/standard-sql/other-statements#export_data_statement) statement. * `EXPORT_MODEL`: [`EXPORT MODEL`](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-export-model) statement. * `LOAD_DATA`: [`LOAD DATA`](/bigquery/docs/reference/standard-sql/other-statements#load_data_statement) statement. * `CALL`: [`CALL`](/bigquery/docs/reference/standard-sql/procedural-language#call) statement.
	Ddlaffectedrowaccesspolicycount string `json:"ddlAffectedRowAccessPolicyCount,omitempty"` // Output only. The number of row access policies affected by a DDL statement. Present only for DROP ALL ROW ACCESS POLICIES queries.
	Performanceinsights PerformanceInsights `json:"performanceInsights,omitempty"` // Performance insights for the job.
	Totalbytesbilled string `json:"totalBytesBilled,omitempty"` // Output only. If the project is configured to use on-demand pricing, then this field contains the total bytes billed for the job. If the project is configured to use flat-rate pricing, then you are not billed for bytes and this field is informational only.
	Dcltargetview TableReference `json:"dclTargetView,omitempty"`
	Undeclaredqueryparameters []QueryParameter `json:"undeclaredQueryParameters,omitempty"` // Output only. GoogleSQL only: list of undeclared query parameters detected during a dry run validation.
	Totalbytesprocessedaccuracy string `json:"totalBytesProcessedAccuracy,omitempty"` // Output only. For dry-run jobs, totalBytesProcessed is an estimate and this field specifies the accuracy of the estimate. Possible values can be: UNKNOWN: accuracy of the estimate is unknown. PRECISE: estimate is precise. LOWER_BOUND: estimate is lower bound of what the query would cost. UPPER_BOUND: estimate is upper bound of what the query would cost.
	Externalservicecosts []ExternalServiceCost `json:"externalServiceCosts,omitempty"` // Output only. Job cost breakdown as bigquery internal cost and external service costs.
	Queryplan []ExplainQueryStage `json:"queryPlan,omitempty"` // Output only. Describes execution plan for the query.
}

// BigtableOptions represents the BigtableOptions schema from the OpenAPI specification
type BigtableOptions struct {
	Columnfamilies []BigtableColumnFamily `json:"columnFamilies,omitempty"` // Optional. List of column families to expose in the table schema along with their types. This list restricts the column families that can be referenced in queries and specifies their value types. You can use this list to do type conversions - see the 'type' field for more details. If you leave this list empty, all column families are present in the table schema and their values are read as BYTES. During a query only the column families referenced in that query are read from Bigtable.
	Ignoreunspecifiedcolumnfamilies bool `json:"ignoreUnspecifiedColumnFamilies,omitempty"` // Optional. If field is true, then the column families that are not specified in columnFamilies list are not exposed in the table schema. Otherwise, they are read with BYTES type values. The default value is false.
	Outputcolumnfamiliesasjson bool `json:"outputColumnFamiliesAsJson,omitempty"` // Optional. If field is true, then each column family will be read as a single JSON column. Otherwise they are read as a repeated cell structure containing timestamp/value tuples. The default value is false.
	Readrowkeyasstring bool `json:"readRowkeyAsString,omitempty"` // Optional. If field is true, then the rowkey column families will be read and converted to string. Otherwise they are read with BYTES type values and users need to manually cast them with CAST if necessary. The default value is false.
}

// JobList represents the JobList schema from the OpenAPI specification
type JobList struct {
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token to request the next page of results.
	Unreachable []string `json:"unreachable,omitempty"` // A list of skipped locations that were unreachable. For more information about BigQuery locations, see: https://cloud.google.com/bigquery/docs/locations. Example: "europe-west5"
	Etag string `json:"etag,omitempty"` // A hash of this page of results.
	Jobs []map[string]interface{} `json:"jobs,omitempty"` // List of jobs that were requested.
	Kind string `json:"kind,omitempty"` // The resource type of the response.
}

// LoadQueryStatistics represents the LoadQueryStatistics schema from the OpenAPI specification
type LoadQueryStatistics struct {
	Outputrows string `json:"outputRows,omitempty"` // Output only. Number of rows imported in a LOAD query. Note that while a LOAD query is in the running state, this value may change.
	Badrecords string `json:"badRecords,omitempty"` // Output only. The number of bad records encountered while processing a LOAD query. Note that if the job has failed because of more bad records encountered than the maximum allowed in the load job configuration, then this number can be less than the total number of bad records present in the input data.
	Bytestransferred string `json:"bytesTransferred,omitempty"` // Output only. This field is deprecated. The number of bytes of source data copied over the network for a `LOAD` query. `transferred_bytes` has the canonical value for physical transferred bytes, which is used for BigQuery Omni billing.
	Inputfilebytes string `json:"inputFileBytes,omitempty"` // Output only. Number of bytes of source data in a LOAD query.
	Inputfiles string `json:"inputFiles,omitempty"` // Output only. Number of source files in a LOAD query.
	Outputbytes string `json:"outputBytes,omitempty"` // Output only. Size of the loaded data in bytes. Note that while a LOAD query is in the running state, this value may change.
}

// StandardSqlDataType represents the StandardSqlDataType schema from the OpenAPI specification
type StandardSqlDataType struct {
	Rangeelementtype StandardSqlDataType `json:"rangeElementType,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
	Structtype StandardSqlStructType `json:"structType,omitempty"` // The representation of a SQL STRUCT type.
	Typekind string `json:"typeKind,omitempty"` // Required. The top level type of this field. Can be any GoogleSQL data type (e.g., "INT64", "DATE", "ARRAY").
	Arrayelementtype StandardSqlDataType `json:"arrayElementType,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
}

// HparamSearchSpaces represents the HparamSearchSpaces schema from the OpenAPI specification
type HparamSearchSpaces struct {
	L2reg DoubleHparamSearchSpace `json:"l2Reg,omitempty"` // Search space for a double hyperparameter.
	Subsample DoubleHparamSearchSpace `json:"subsample,omitempty"` // Search space for a double hyperparameter.
	Learnrate DoubleHparamSearchSpace `json:"learnRate,omitempty"` // Search space for a double hyperparameter.
	Numclusters IntHparamSearchSpace `json:"numClusters,omitempty"` // Search space for an int hyperparameter.
	Activationfn StringHparamSearchSpace `json:"activationFn,omitempty"` // Search space for string and enum.
	Numparalleltree IntHparamSearchSpace `json:"numParallelTree,omitempty"` // Search space for an int hyperparameter.
	Maxtreedepth IntHparamSearchSpace `json:"maxTreeDepth,omitempty"` // Search space for an int hyperparameter.
	Batchsize IntHparamSearchSpace `json:"batchSize,omitempty"` // Search space for an int hyperparameter.
	Hiddenunits IntArrayHparamSearchSpace `json:"hiddenUnits,omitempty"` // Search space for int array.
	L1reg DoubleHparamSearchSpace `json:"l1Reg,omitempty"` // Search space for a double hyperparameter.
	Numfactors IntHparamSearchSpace `json:"numFactors,omitempty"` // Search space for an int hyperparameter.
	Dropout DoubleHparamSearchSpace `json:"dropout,omitempty"` // Search space for a double hyperparameter.
	Mintreechildweight IntHparamSearchSpace `json:"minTreeChildWeight,omitempty"` // Search space for an int hyperparameter.
	Optimizer StringHparamSearchSpace `json:"optimizer,omitempty"` // Search space for string and enum.
	Minsplitloss DoubleHparamSearchSpace `json:"minSplitLoss,omitempty"` // Search space for a double hyperparameter.
	Walsalpha DoubleHparamSearchSpace `json:"walsAlpha,omitempty"` // Search space for a double hyperparameter.
	Dartnormalizetype StringHparamSearchSpace `json:"dartNormalizeType,omitempty"` // Search space for string and enum.
	Colsamplebynode DoubleHparamSearchSpace `json:"colsampleBynode,omitempty"` // Search space for a double hyperparameter.
	Colsamplebytree DoubleHparamSearchSpace `json:"colsampleBytree,omitempty"` // Search space for a double hyperparameter.
	Treemethod StringHparamSearchSpace `json:"treeMethod,omitempty"` // Search space for string and enum.
	Boostertype StringHparamSearchSpace `json:"boosterType,omitempty"` // Search space for string and enum.
	Colsamplebylevel DoubleHparamSearchSpace `json:"colsampleBylevel,omitempty"` // Search space for a double hyperparameter.
}

// SparkStatistics represents the SparkStatistics schema from the OpenAPI specification
type SparkStatistics struct {
	Gcsstagingbucket string `json:"gcsStagingBucket,omitempty"` // Output only. The Google Cloud Storage bucket that is used as the default filesystem by the Spark application. This fields is only filled when the Spark procedure uses the INVOKER security mode. It is inferred from the system variable @@spark_proc_properties.staging_bucket if it is provided. Otherwise, BigQuery creates a default staging bucket for the job and returns the bucket name in this field. Example: * `gs://[bucket_name]`
	Kmskeyname string `json:"kmsKeyName,omitempty"` // Output only. The Cloud KMS encryption key that is used to protect the resources created by the Spark job. If the Spark procedure uses DEFINER security mode, the Cloud KMS key is inferred from the Spark connection associated with the procedure if it is provided. Otherwise the key is inferred from the default key of the Spark connection's project if the CMEK organization policy is enforced. If the Spark procedure uses INVOKER security mode, the Cloud KMS encryption key is inferred from the system variable @@spark_proc_properties.kms_key_name if it is provided. Otherwise, the key is inferred fromt he default key of the BigQuery job's project if the CMEK organization policy is enforced. Example: * `projects/[kms_project_id]/locations/[region]/keyRings/[key_region]/cryptoKeys/[key]`
	Logginginfo SparkLoggingInfo `json:"loggingInfo,omitempty"` // Spark job logs can be filtered by these fields in Cloud Logging.
	Sparkjobid string `json:"sparkJobId,omitempty"` // Output only. Spark job ID if a Spark job is created successfully.
	Sparkjoblocation string `json:"sparkJobLocation,omitempty"` // Output only. Location where the Spark job is executed. A location is selected by BigQueury for jobs configured to run in a multi-region.
	Endpoints map[string]interface{} `json:"endpoints,omitempty"` // Output only. Endpoints returned from Dataproc. Key list: - history_server_endpoint: A link to Spark job UI.
}

// ProjectList represents the ProjectList schema from the OpenAPI specification
type ProjectList struct {
	Etag string `json:"etag,omitempty"` // A hash of the page of results.
	Kind string `json:"kind,omitempty"` // The resource type of the response.
	Nextpagetoken string `json:"nextPageToken,omitempty"` // Use this token to request the next page of results.
	Projects []map[string]interface{} `json:"projects,omitempty"` // Projects to which the user has at least READ access.
	Totalitems int `json:"totalItems,omitempty"` // The total number of projects in the page. A wrapper is used here because the field should still be in the response when the value is 0.
}

// BiEngineStatistics represents the BiEngineStatistics schema from the OpenAPI specification
type BiEngineStatistics struct {
	Bienginereasons []BiEngineReason `json:"biEngineReasons,omitempty"` // In case of DISABLED or PARTIAL bi_engine_mode, these contain the explanatory reasons as to why BI Engine could not accelerate. In case the full query was accelerated, this field is not populated.
	Accelerationmode string `json:"accelerationMode,omitempty"` // Output only. Specifies which mode of BI Engine acceleration was performed (if any).
	Bienginemode string `json:"biEngineMode,omitempty"` // Output only. Specifies which mode of BI Engine acceleration was performed (if any).
}

// MaterializedViewStatus represents the MaterializedViewStatus schema from the OpenAPI specification
type MaterializedViewStatus struct {
	Lastrefreshstatus ErrorProto `json:"lastRefreshStatus,omitempty"` // Error details.
	Refreshwatermark string `json:"refreshWatermark,omitempty"` // Output only. Refresh watermark of materialized view. The base tables' data were collected into the materialized view cache until this time.
}

// ClusteringMetrics represents the ClusteringMetrics schema from the OpenAPI specification
type ClusteringMetrics struct {
	Daviesbouldinindex float64 `json:"daviesBouldinIndex,omitempty"` // Davies-Bouldin index.
	Meansquareddistance float64 `json:"meanSquaredDistance,omitempty"` // Mean of squared distances between each sample to its cluster centroid.
	Clusters []Cluster `json:"clusters,omitempty"` // Information for all clusters.
}

// ModelExtractOptions represents the ModelExtractOptions schema from the OpenAPI specification
type ModelExtractOptions struct {
	Trialid string `json:"trialId,omitempty"` // The 1-based ID of the trial to be exported from a hyperparameter tuning model. If not specified, the trial with id = [Model](/bigquery/docs/reference/rest/v2/models#resource:-model).defaultTrialId is exported. This field is ignored for models not trained with hyperparameter tuning.
}

// TrainingRun represents the TrainingRun schema from the OpenAPI specification
type TrainingRun struct {
	Datasplitresult DataSplitResult `json:"dataSplitResult,omitempty"` // Data split result. This contains references to the training and evaluation data tables that were used to train the model.
	Trainingstarttime string `json:"trainingStartTime,omitempty"` // Output only. The start time of this training run, in milliseconds since epoch.
	Classlevelglobalexplanations []GlobalExplanation `json:"classLevelGlobalExplanations,omitempty"` // Output only. Global explanation contains the explanation of top features on the class level. Applies to classification models only.
	Results []IterationResult `json:"results,omitempty"` // Output only. Output of each iteration run, results.size() <= max_iterations.
	Starttime string `json:"startTime,omitempty"` // Output only. The start time of this training run.
	Trainingoptions TrainingOptions `json:"trainingOptions,omitempty"` // Options used in model training.
	Evaluationmetrics EvaluationMetrics `json:"evaluationMetrics,omitempty"` // Evaluation metrics of a model. These are either computed on all training data or just the eval data based on whether eval data was used during training. These are not present for imported models.
	Modellevelglobalexplanation GlobalExplanation `json:"modelLevelGlobalExplanation,omitempty"` // Global explanations containing the top most important features after training.
	Vertexaimodelid string `json:"vertexAiModelId,omitempty"` // The model id in the [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction) for this training run.
	Vertexaimodelversion string `json:"vertexAiModelVersion,omitempty"` // Output only. The model version in the [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction) for this training run.
}

// ScriptOptions represents the ScriptOptions schema from the OpenAPI specification
type ScriptOptions struct {
	Keyresultstatement string `json:"keyResultStatement,omitempty"` // Determines which statement in the script represents the "key result", used to populate the schema and query results of the script job. Default is LAST.
	Statementbytebudget string `json:"statementByteBudget,omitempty"` // Limit on the number of bytes billed per statement. Exceeding this budget results in an error.
	Statementtimeoutms string `json:"statementTimeoutMs,omitempty"` // Timeout period for each statement in a script.
}

// ArimaForecastingMetrics represents the ArimaForecastingMetrics schema from the OpenAPI specification
type ArimaForecastingMetrics struct {
	Arimafittingmetrics []ArimaFittingMetrics `json:"arimaFittingMetrics,omitempty"` // Arima model fitting metrics.
	Arimasinglemodelforecastingmetrics []ArimaSingleModelForecastingMetrics `json:"arimaSingleModelForecastingMetrics,omitempty"` // Repeated as there can be many metric sets (one for each model) in auto-arima and the large-scale case.
	Hasdrift []bool `json:"hasDrift,omitempty"` // Whether Arima model fitted with drift or not. It is always false when d is not 1.
	Nonseasonalorder []ArimaOrder `json:"nonSeasonalOrder,omitempty"` // Non-seasonal order.
	Seasonalperiods []string `json:"seasonalPeriods,omitempty"` // Seasonal periods. Repeated because multiple periods are supported for one time series.
	Timeseriesid []string `json:"timeSeriesId,omitempty"` // Id to differentiate different time series for the large-scale case.
}

// BigtableColumn represents the BigtableColumn schema from the OpenAPI specification
type BigtableColumn struct {
	Onlyreadlatest bool `json:"onlyReadLatest,omitempty"` // Optional. If this is set, only the latest version of value in this column are exposed. 'onlyReadLatest' can also be set at the column family level. However, the setting at this level takes precedence if 'onlyReadLatest' is set at both levels.
	Qualifierencoded string `json:"qualifierEncoded,omitempty"` // [Required] Qualifier of the column. Columns in the parent column family that has this exact qualifier are exposed as . field. If the qualifier is valid UTF-8 string, it can be specified in the qualifier_string field. Otherwise, a base-64 encoded value must be set to qualifier_encoded. The column field name is the same as the column qualifier. However, if the qualifier is not a valid BigQuery field identifier i.e. does not match a-zA-Z*, a valid identifier must be provided as field_name.
	Qualifierstring string `json:"qualifierString,omitempty"` // Qualifier string.
	TypeField string `json:"type,omitempty"` // Optional. The type to convert the value in cells of this column. The values are expected to be encoded using HBase Bytes.toBytes function when using the BINARY encoding value. Following BigQuery types are allowed (case-sensitive): * BYTES * STRING * INTEGER * FLOAT * BOOLEAN * JSON Default type is BYTES. 'type' can also be set at the column family level. However, the setting at this level takes precedence if 'type' is set at both levels.
	Encoding string `json:"encoding,omitempty"` // Optional. The encoding of the values when the type is not STRING. Acceptable encoding values are: TEXT - indicates values are alphanumeric text strings. BINARY - indicates values are encoded using HBase Bytes.toBytes family of functions. 'encoding' can also be set at the column family level. However, the setting at this level takes precedence if 'encoding' is set at both levels.
	Fieldname string `json:"fieldName,omitempty"` // Optional. If the qualifier is not a valid BigQuery field identifier i.e. does not match a-zA-Z*, a valid identifier must be provided as the column field name and is used as field name in queries.
}

// Streamingbuffer represents the Streamingbuffer schema from the OpenAPI specification
type Streamingbuffer struct {
	Estimatedbytes string `json:"estimatedBytes,omitempty"` // Output only. A lower-bound estimate of the number of bytes currently in the streaming buffer.
	Estimatedrows string `json:"estimatedRows,omitempty"` // Output only. A lower-bound estimate of the number of rows currently in the streaming buffer.
	Oldestentrytime string `json:"oldestEntryTime,omitempty"` // Output only. Contains the timestamp of the oldest entry in the streaming buffer, in milliseconds since the epoch, if the streaming buffer is available.
}

// UndeleteDatasetRequest represents the UndeleteDatasetRequest schema from the OpenAPI specification
type UndeleteDatasetRequest struct {
	Deletiontime string `json:"deletionTime,omitempty"` // Optional. The exact time when the dataset was deleted. If not specified, it will undelete the most recently deleted version.
}

// JobReference represents the JobReference schema from the OpenAPI specification
type JobReference struct {
	Jobid string `json:"jobId,omitempty"` // Required. The ID of the job. The ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_), or dashes (-). The maximum length is 1,024 characters.
	Location string `json:"location,omitempty"` // Optional. The geographic location of the job. The default value is US. For more information about BigQuery locations, see: https://cloud.google.com/bigquery/docs/locations
	Projectid string `json:"projectId,omitempty"` // Required. The ID of the project containing this job.
}

// RemoteFunctionOptions represents the RemoteFunctionOptions schema from the OpenAPI specification
type RemoteFunctionOptions struct {
	Connection string `json:"connection,omitempty"` // Fully qualified name of the user-provided connection object which holds the authentication information to send requests to the remote service. Format: ```"projects/{projectId}/locations/{locationId}/connections/{connectionId}"```
	Endpoint string `json:"endpoint,omitempty"` // Endpoint of the user-provided remote service, e.g. ```https://us-east1-my_gcf_project.cloudfunctions.net/remote_add```
	Maxbatchingrows string `json:"maxBatchingRows,omitempty"` // Max number of rows in each batch sent to the remote service. If absent or if 0, BigQuery dynamically decides the number of rows in a batch.
	Userdefinedcontext map[string]interface{} `json:"userDefinedContext,omitempty"` // User-defined context as a set of key/value pairs, which will be sent as function invocation context together with batched arguments in the requests to the remote service. The total number of bytes of keys and values must be less than 8KB.
}

// RowAccessPolicyReference represents the RowAccessPolicyReference schema from the OpenAPI specification
type RowAccessPolicyReference struct {
	Datasetid string `json:"datasetId,omitempty"` // Required. The ID of the dataset containing this row access policy.
	Policyid string `json:"policyId,omitempty"` // Required. The ID of the row access policy. The ID must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_). The maximum length is 256 characters.
	Projectid string `json:"projectId,omitempty"` // Required. The ID of the project containing this row access policy.
	Tableid string `json:"tableId,omitempty"` // Required. The ID of the table containing this row access policy.
}

// DimensionalityReductionMetrics represents the DimensionalityReductionMetrics schema from the OpenAPI specification
type DimensionalityReductionMetrics struct {
	Totalexplainedvarianceratio float64 `json:"totalExplainedVarianceRatio,omitempty"` // Total percentage of variance explained by the selected principal components.
}

// SparkLoggingInfo represents the SparkLoggingInfo schema from the OpenAPI specification
type SparkLoggingInfo struct {
	Resourcetype string `json:"resourceType,omitempty"` // Output only. Resource type used for logging.
	Projectid string `json:"projectId,omitempty"` // Output only. Project ID where the Spark logs were written.
}

// TableList represents the TableList schema from the OpenAPI specification
type TableList struct {
	Totalitems int `json:"totalItems,omitempty"` // The total number of tables in the dataset.
	Etag string `json:"etag,omitempty"` // A hash of this page of results.
	Kind string `json:"kind,omitempty"` // The type of list.
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token to request the next page of results.
	Tables []map[string]interface{} `json:"tables,omitempty"` // Tables in the requested dataset.
}

// ModelReference represents the ModelReference schema from the OpenAPI specification
type ModelReference struct {
	Modelid string `json:"modelId,omitempty"` // Required. The ID of the model. The ID must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_). The maximum length is 1,024 characters.
	Projectid string `json:"projectId,omitempty"` // Required. The ID of the project containing this model.
	Datasetid string `json:"datasetId,omitempty"` // Required. The ID of the dataset containing this model.
}

// BqmlTrainingRun represents the BqmlTrainingRun schema from the OpenAPI specification
type BqmlTrainingRun struct {
	State string `json:"state,omitempty"` // Deprecated.
	Trainingoptions map[string]interface{} `json:"trainingOptions,omitempty"` // Deprecated.
	Iterationresults []BqmlIterationResult `json:"iterationResults,omitempty"` // Deprecated.
	Starttime string `json:"startTime,omitempty"` // Deprecated.
}

// DatasetAccessEntry represents the DatasetAccessEntry schema from the OpenAPI specification
type DatasetAccessEntry struct {
	Dataset DatasetReference `json:"dataset,omitempty"`
	Targettypes []string `json:"targetTypes,omitempty"` // Which resources in the dataset this entry applies to. Currently, only views are supported, but additional target types may be added in the future.
}

// ExplainQueryStep represents the ExplainQueryStep schema from the OpenAPI specification
type ExplainQueryStep struct {
	Kind string `json:"kind,omitempty"` // Machine-readable operation type.
	Substeps []string `json:"substeps,omitempty"` // Human-readable description of the step(s).
}

// DataSplitResult represents the DataSplitResult schema from the OpenAPI specification
type DataSplitResult struct {
	Testtable TableReference `json:"testTable,omitempty"`
	Trainingtable TableReference `json:"trainingTable,omitempty"`
	Evaluationtable TableReference `json:"evaluationTable,omitempty"`
}

// HighCardinalityJoin represents the HighCardinalityJoin schema from the OpenAPI specification
type HighCardinalityJoin struct {
	Leftrows string `json:"leftRows,omitempty"` // Output only. Count of left input rows.
	Outputrows string `json:"outputRows,omitempty"` // Output only. Count of the output rows.
	Rightrows string `json:"rightRows,omitempty"` // Output only. Count of right input rows.
	Stepindex int `json:"stepIndex,omitempty"` // Output only. The index of the join operator in the ExplainQueryStep lists.
}

// LocationMetadata represents the LocationMetadata schema from the OpenAPI specification
type LocationMetadata struct {
	Legacylocationid string `json:"legacyLocationId,omitempty"` // The legacy BigQuery location ID, e.g. “EU” for the “europe” location. This is for any API consumers that need the legacy “US” and “EU” locations.
}

// EncryptionConfiguration represents the EncryptionConfiguration schema from the OpenAPI specification
type EncryptionConfiguration struct {
	Kmskeyname string `json:"kmsKeyName,omitempty"` // Optional. Describes the Cloud KMS encryption key that will be used to protect destination BigQuery table. The BigQuery Service Account associated with your project requires access to this encryption key.
}

// BinaryConfusionMatrix represents the BinaryConfusionMatrix schema from the OpenAPI specification
type BinaryConfusionMatrix struct {
	Precision float64 `json:"precision,omitempty"` // The fraction of actual positive predictions that had positive actual labels.
	Truenegatives string `json:"trueNegatives,omitempty"` // Number of true samples predicted as false.
	Truepositives string `json:"truePositives,omitempty"` // Number of true samples predicted as true.
	Recall float64 `json:"recall,omitempty"` // The fraction of actual positive labels that were given a positive prediction.
	Falsepositives string `json:"falsePositives,omitempty"` // Number of false samples predicted as true.
	Positiveclassthreshold float64 `json:"positiveClassThreshold,omitempty"` // Threshold value used when computing each of the following metric.
	F1score float64 `json:"f1Score,omitempty"` // The equally weighted average of recall and precision.
	Accuracy float64 `json:"accuracy,omitempty"` // The fraction of predictions given the correct label.
	Falsenegatives string `json:"falseNegatives,omitempty"` // Number of false samples predicted as false.
}

// ArimaSingleModelForecastingMetrics represents the ArimaSingleModelForecastingMetrics schema from the OpenAPI specification
type ArimaSingleModelForecastingMetrics struct {
	Hasspikesanddips bool `json:"hasSpikesAndDips,omitempty"` // If true, spikes_and_dips is a part of time series decomposition result.
	Nonseasonalorder ArimaOrder `json:"nonSeasonalOrder,omitempty"` // Arima order, can be used for both non-seasonal and seasonal parts.
	Timeseriesid string `json:"timeSeriesId,omitempty"` // The time_series_id value for this time series. It will be one of the unique values from the time_series_id_column specified during ARIMA model training. Only present when time_series_id_column training option was used.
	Timeseriesids []string `json:"timeSeriesIds,omitempty"` // The tuple of time_series_ids identifying this time series. It will be one of the unique tuples of values present in the time_series_id_columns specified during ARIMA model training. Only present when time_series_id_columns training option was used and the order of values here are same as the order of time_series_id_columns.
	Seasonalperiods []string `json:"seasonalPeriods,omitempty"` // Seasonal periods. Repeated because multiple periods are supported for one time series.
	Arimafittingmetrics ArimaFittingMetrics `json:"arimaFittingMetrics,omitempty"` // ARIMA model fitting metrics.
	Hasholidayeffect bool `json:"hasHolidayEffect,omitempty"` // If true, holiday_effect is a part of time series decomposition result.
	Hasstepchanges bool `json:"hasStepChanges,omitempty"` // If true, step_changes is a part of time series decomposition result.
	Hasdrift bool `json:"hasDrift,omitempty"` // Is arima model fitted with drift or not. It is always false when d is not 1.
}

// RowAccessPolicy represents the RowAccessPolicy schema from the OpenAPI specification
type RowAccessPolicy struct {
	Lastmodifiedtime string `json:"lastModifiedTime,omitempty"` // Output only. The time when this row access policy was last modified, in milliseconds since the epoch.
	Rowaccesspolicyreference RowAccessPolicyReference `json:"rowAccessPolicyReference,omitempty"` // Id path of a row access policy.
	Creationtime string `json:"creationTime,omitempty"` // Output only. The time when this row access policy was created, in milliseconds since the epoch.
	Etag string `json:"etag,omitempty"` // Output only. A hash of this resource.
	Filterpredicate string `json:"filterPredicate,omitempty"` // Required. A SQL boolean expression that represents the rows defined by this row access policy, similar to the boolean expression in a WHERE clause of a SELECT query on a table. References to other tables, routines, and temporary functions are not supported. Examples: region="EU" date_field = CAST('2019-9-27' as DATE) nullable_field is not NULL numeric_field BETWEEN 1.0 AND 5.0
}

// JobStatistics3 represents the JobStatistics3 schema from the OpenAPI specification
type JobStatistics3 struct {
	Outputrows string `json:"outputRows,omitempty"` // Output only. Number of rows imported in a load job. Note that while an import job is in the running state, this value may change.
	Timeline []QueryTimelineSample `json:"timeline,omitempty"` // Output only. Describes a timeline of job execution.
	Badrecords string `json:"badRecords,omitempty"` // Output only. The number of bad records encountered. Note that if the job has failed because of more bad records encountered than the maximum allowed in the load job configuration, then this number can be less than the total number of bad records present in the input data.
	Inputfilebytes string `json:"inputFileBytes,omitempty"` // Output only. Number of bytes of source data in a load job.
	Inputfiles string `json:"inputFiles,omitempty"` // Output only. Number of source files in a load job.
	Outputbytes string `json:"outputBytes,omitempty"` // Output only. Size of the loaded data in bytes. Note that while a load job is in the running state, this value may change.
}

// Model represents the Model schema from the OpenAPI specification
type Model struct {
	Expirationtime string `json:"expirationTime,omitempty"` // Optional. The time when this model expires, in milliseconds since the epoch. If not present, the model will persist indefinitely. Expired models will be deleted and their storage reclaimed. The defaultTableExpirationMs property of the encapsulating dataset can be used to set a default expirationTime on newly created models.
	Featurecolumns []StandardSqlField `json:"featureColumns,omitempty"` // Output only. Input feature columns for the model inference. If the model is trained with TRANSFORM clause, these are the input of the TRANSFORM clause.
	Location string `json:"location,omitempty"` // Output only. The geographic location where the model resides. This value is inherited from the dataset.
	Modelreference ModelReference `json:"modelReference,omitempty"` // Id path of a model.
	Modeltype string `json:"modelType,omitempty"` // Output only. Type of the model resource.
	Remotemodelinfo RemoteModelInfo `json:"remoteModelInfo,omitempty"` // Remote Model Info
	Hparamtrials []HparamTuningTrial `json:"hparamTrials,omitempty"` // Output only. Trials of a [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) model sorted by trial_id.
	Labelcolumns []StandardSqlField `json:"labelColumns,omitempty"` // Output only. Label columns that were used to train this model. The output of the model will have a "predicted_" prefix to these columns.
	Creationtime string `json:"creationTime,omitempty"` // Output only. The time when this model was created, in millisecs since the epoch.
	Description string `json:"description,omitempty"` // Optional. A user-friendly description of this model.
	Lastmodifiedtime string `json:"lastModifiedTime,omitempty"` // Output only. The time when this model was last modified, in millisecs since the epoch.
	Defaulttrialid string `json:"defaultTrialId,omitempty"` // Output only. The default trial_id to use in TVFs when the trial_id is not passed in. For single-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, this is the best trial ID. For multi-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, this is the smallest trial ID among all Pareto optimal trials.
	Encryptionconfiguration EncryptionConfiguration `json:"encryptionConfiguration,omitempty"`
	Trainingruns []TrainingRun `json:"trainingRuns,omitempty"` // Information for all training runs in increasing order of start_time.
	Etag string `json:"etag,omitempty"` // Output only. A hash of this resource.
	Friendlyname string `json:"friendlyName,omitempty"` // Optional. A descriptive name for this model.
	Besttrialid string `json:"bestTrialId,omitempty"` // The best trial_id across all training runs.
	Transformcolumns []TransformColumn `json:"transformColumns,omitempty"` // Output only. This field will be populated if a TRANSFORM clause was used to train a model. TRANSFORM clause (if used) takes feature_columns as input and outputs transform_columns. transform_columns then are used to train the model.
	Optimaltrialids []string `json:"optimalTrialIds,omitempty"` // Output only. For single-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, it only contains the best trial. For multi-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, it contains all Pareto optimal trials sorted by trial_id.
	Hparamsearchspaces HparamSearchSpaces `json:"hparamSearchSpaces,omitempty"` // Hyperparameter search spaces. These should be a subset of training_options.
	Labels map[string]interface{} `json:"labels,omitempty"` // The labels associated with this model. You can use these to organize and group your models. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label values are optional. Label keys must start with a letter and each label in the list must have a different key.
}

// GetServiceAccountResponse represents the GetServiceAccountResponse schema from the OpenAPI specification
type GetServiceAccountResponse struct {
	Email string `json:"email,omitempty"` // The service account email address.
	Kind string `json:"kind,omitempty"` // The resource type of the response.
}

// BiEngineReason represents the BiEngineReason schema from the OpenAPI specification
type BiEngineReason struct {
	Code string `json:"code,omitempty"` // Output only. High-level BI Engine reason for partial or disabled acceleration
	Message string `json:"message,omitempty"` // Output only. Free form human-readable reason for partial or disabled acceleration.
}

// BigtableColumnFamily represents the BigtableColumnFamily schema from the OpenAPI specification
type BigtableColumnFamily struct {
	Onlyreadlatest bool `json:"onlyReadLatest,omitempty"` // Optional. If this is set only the latest version of value are exposed for all columns in this column family. This can be overridden for a specific column by listing that column in 'columns' and specifying a different setting for that column.
	TypeField string `json:"type,omitempty"` // Optional. The type to convert the value in cells of this column family. The values are expected to be encoded using HBase Bytes.toBytes function when using the BINARY encoding value. Following BigQuery types are allowed (case-sensitive): * BYTES * STRING * INTEGER * FLOAT * BOOLEAN * JSON Default type is BYTES. This can be overridden for a specific column by listing that column in 'columns' and specifying a type for it.
	Columns []BigtableColumn `json:"columns,omitempty"` // Optional. Lists of columns that should be exposed as individual fields as opposed to a list of (column name, value) pairs. All columns whose qualifier matches a qualifier in this list can be accessed as .. Other columns can be accessed as a list through .Column field.
	Encoding string `json:"encoding,omitempty"` // Optional. The encoding of the values when the type is not STRING. Acceptable encoding values are: TEXT - indicates values are alphanumeric text strings. BINARY - indicates values are encoded using HBase Bytes.toBytes family of functions. This can be overridden for a specific column by listing that column in 'columns' and specifying an encoding for it.
	Familyid string `json:"familyId,omitempty"` // Identifier of the column family.
}

// DataMaskingStatistics represents the DataMaskingStatistics schema from the OpenAPI specification
type DataMaskingStatistics struct {
	Datamaskingapplied bool `json:"dataMaskingApplied,omitempty"` // Whether any accessed data was protected by the data masking.
}

// EvaluationMetrics represents the EvaluationMetrics schema from the OpenAPI specification
type EvaluationMetrics struct {
	Binaryclassificationmetrics BinaryClassificationMetrics `json:"binaryClassificationMetrics,omitempty"` // Evaluation metrics for binary classification/classifier models.
	Clusteringmetrics ClusteringMetrics `json:"clusteringMetrics,omitempty"` // Evaluation metrics for clustering models.
	Dimensionalityreductionmetrics DimensionalityReductionMetrics `json:"dimensionalityReductionMetrics,omitempty"` // Model evaluation metrics for dimensionality reduction models.
	Multiclassclassificationmetrics MultiClassClassificationMetrics `json:"multiClassClassificationMetrics,omitempty"` // Evaluation metrics for multi-class classification/classifier models.
	Rankingmetrics RankingMetrics `json:"rankingMetrics,omitempty"` // Evaluation metrics used by weighted-ALS models specified by feedback_type=implicit.
	Regressionmetrics RegressionMetrics `json:"regressionMetrics,omitempty"` // Evaluation metrics for regression and explicit feedback type matrix factorization models.
	Arimaforecastingmetrics ArimaForecastingMetrics `json:"arimaForecastingMetrics,omitempty"` // Model evaluation metrics for ARIMA forecasting models.
}

// TestIamPermissionsResponse represents the TestIamPermissionsResponse schema from the OpenAPI specification
type TestIamPermissionsResponse struct {
	Permissions []string `json:"permissions,omitempty"` // A subset of `TestPermissionsRequest.permissions` that the caller is allowed.
}

// Entry represents the Entry schema from the OpenAPI specification
type Entry struct {
	Itemcount string `json:"itemCount,omitempty"` // Number of items being predicted as this label.
	Predictedlabel string `json:"predictedLabel,omitempty"` // The predicted label. For confidence_threshold > 0, we will also add an entry indicating the number of items under the confidence threshold.
}

// IndexUnusedReason represents the IndexUnusedReason schema from the OpenAPI specification
type IndexUnusedReason struct {
	Basetable TableReference `json:"baseTable,omitempty"`
	Code string `json:"code,omitempty"` // Specifies the high-level reason for the scenario when no search index was used.
	Indexname string `json:"indexName,omitempty"` // Specifies the name of the unused search index, if available.
	Message string `json:"message,omitempty"` // Free form human-readable reason for the scenario when no search index was used.
}

// LinkedDatasetSource represents the LinkedDatasetSource schema from the OpenAPI specification
type LinkedDatasetSource struct {
	Sourcedataset DatasetReference `json:"sourceDataset,omitempty"`
}

// Explanation represents the Explanation schema from the OpenAPI specification
type Explanation struct {
	Attribution float64 `json:"attribution,omitempty"` // Attribution of feature.
	Featurename string `json:"featureName,omitempty"` // The full feature name. For non-numerical features, will be formatted like `.`. Overall size of feature name will always be truncated to first 120 characters.
}

// Dataset represents the Dataset schema from the OpenAPI specification
type Dataset struct {
	Linkeddatasetsource LinkedDatasetSource `json:"linkedDatasetSource,omitempty"` // A dataset source type which refers to another BigQuery dataset.
	Datasetreference DatasetReference `json:"datasetReference,omitempty"`
	Satisfiespzi bool `json:"satisfiesPzi,omitempty"` // Output only. Reserved for future use.
	TypeField string `json:"type,omitempty"` // Output only. Same as `type` in `ListFormatDataset`. The type of the dataset, one of: * DEFAULT - only accessible by owner and authorized accounts, * PUBLIC - accessible by everyone, * LINKED - linked dataset, * EXTERNAL - dataset with definition in external metadata catalog. -- *BIGLAKE_METASTORE - dataset that references a database created in BigLakeMetastore service. --
	Id string `json:"id,omitempty"` // Output only. The fully-qualified unique name of the dataset in the format projectId:datasetId. The dataset name without the project name is given in the datasetId field. When creating a new dataset, leave this field blank, and instead specify the datasetId field.
	Creationtime string `json:"creationTime,omitempty"` // Output only. The time when this dataset was created, in milliseconds since the epoch.
	Kind string `json:"kind,omitempty"` // Output only. The resource type.
	Lastmodifiedtime string `json:"lastModifiedTime,omitempty"` // Output only. The date when this dataset was last modified, in milliseconds since the epoch.
	Defaulttableexpirationms string `json:"defaultTableExpirationMs,omitempty"` // Optional. The default lifetime of all tables in the dataset, in milliseconds. The minimum lifetime value is 3600000 milliseconds (one hour). To clear an existing default expiration with a PATCH request, set to 0. Once this property is set, all newly-created tables in the dataset will have an expirationTime property set to the creation time plus the value in this property, and changing the value will only affect new tables, not existing ones. When the expirationTime for a given table is reached, that table will be deleted automatically. If a table's expirationTime is modified or removed before the table expires, or if you provide an explicit expirationTime when creating a table, that value takes precedence over the default expiration time indicated by this property.
	Tags []map[string]interface{} `json:"tags,omitempty"` // Output only. Tags for the Dataset.
	Defaultpartitionexpirationms string `json:"defaultPartitionExpirationMs,omitempty"` // This default partition expiration, expressed in milliseconds. When new time-partitioned tables are created in a dataset where this property is set, the table will inherit this value, propagated as the `TimePartitioning.expirationMs` property on the new table. If you set `TimePartitioning.expirationMs` explicitly when creating a table, the `defaultPartitionExpirationMs` of the containing dataset is ignored. When creating a partitioned table, if `defaultPartitionExpirationMs` is set, the `defaultTableExpirationMs` value is ignored and the table will not be inherit a table expiration deadline.
	Friendlyname string `json:"friendlyName,omitempty"` // Optional. A descriptive name for the dataset.
	Maxtimetravelhours string `json:"maxTimeTravelHours,omitempty"` // Optional. Defines the time travel window in hours. The value can be from 48 to 168 hours (2 to 7 days). The default value is 168 hours if this is not set.
	Defaultroundingmode string `json:"defaultRoundingMode,omitempty"` // Optional. Defines the default rounding mode specification of new tables created within this dataset. During table creation, if this field is specified, the table within this dataset will inherit the default rounding mode of the dataset. Setting the default rounding mode on a table overrides this option. Existing tables in the dataset are unaffected. If columns are defined during that table creation, they will immediately inherit the table's default rounding mode, unless otherwise specified.
	Labels map[string]interface{} `json:"labels,omitempty"` // The labels associated with this dataset. You can use these to organize and group your datasets. You can set this property when inserting or updating a dataset. See Creating and Updating Dataset Labels for more information.
	Iscaseinsensitive bool `json:"isCaseInsensitive,omitempty"` // Optional. TRUE if the dataset and its table names are case-insensitive, otherwise FALSE. By default, this is FALSE, which means the dataset and its table names are case-sensitive. This field does not affect routine references.
	Satisfiespzs bool `json:"satisfiesPzs,omitempty"` // Output only. Reserved for future use.
	Storagebillingmodel string `json:"storageBillingModel,omitempty"` // Optional. Updates storage_billing_model for the dataset.
	Description string `json:"description,omitempty"` // Optional. A user-friendly description of the dataset.
	Etag string `json:"etag,omitempty"` // Output only. A hash of the resource.
	Defaultcollation string `json:"defaultCollation,omitempty"` // Optional. Defines the default collation specification of future tables created in the dataset. If a table is created in this dataset without table-level default collation, then the table inherits the dataset default collation, which is applied to the string fields that do not have explicit collation specified. A change to this field affects only tables created afterwards, and does not alter the existing tables. The following values are supported: * 'und:ci': undetermined locale, case insensitive. * '': empty string. Default to case-sensitive behavior.
	Access []map[string]interface{} `json:"access,omitempty"` // Optional. An array of objects that define dataset access for one or more entities. You can set this property when inserting or updating a dataset in order to control who is allowed to access the data. If unspecified at dataset creation time, BigQuery adds default dataset access for the following entities: access.specialGroup: projectReaders; access.role: READER; access.specialGroup: projectWriters; access.role: WRITER; access.specialGroup: projectOwners; access.role: OWNER; access.userByEmail: [dataset creator email]; access.role: OWNER;
	Externaldatasetreference ExternalDatasetReference `json:"externalDatasetReference,omitempty"` // Configures the access a dataset defined in an external metadata storage.
	Location string `json:"location,omitempty"` // The geographic location where the dataset should reside. See https://cloud.google.com/bigquery/docs/locations for supported locations.
	Selflink string `json:"selfLink,omitempty"` // Output only. A URL that can be used to access the resource again. You can use this URL in Get or Update requests to the resource.
	Defaultencryptionconfiguration EncryptionConfiguration `json:"defaultEncryptionConfiguration,omitempty"`
}

// ErrorProto represents the ErrorProto schema from the OpenAPI specification
type ErrorProto struct {
	Message string `json:"message,omitempty"` // A human-readable description of the error.
	Reason string `json:"reason,omitempty"` // A short error code that summarizes the error.
	Debuginfo string `json:"debugInfo,omitempty"` // Debugging information. This property is internal to Google and should not be used.
	Location string `json:"location,omitempty"` // Specifies where the error occurred, if present.
}

// ExternalDatasetReference represents the ExternalDatasetReference schema from the OpenAPI specification
type ExternalDatasetReference struct {
	Connection string `json:"connection,omitempty"` // Required. The connection id that is used to access the external_source. Format: projects/{project_id}/locations/{location_id}/connections/{connection_id}
	Externalsource string `json:"externalSource,omitempty"` // Required. External source that backs this dataset.
}

// JobCancelResponse represents the JobCancelResponse schema from the OpenAPI specification
type JobCancelResponse struct {
	Job Job `json:"job,omitempty"`
	Kind string `json:"kind,omitempty"` // The resource type of the response.
}

// RemoteModelInfo represents the RemoteModelInfo schema from the OpenAPI specification
type RemoteModelInfo struct {
	Maxbatchingrows string `json:"maxBatchingRows,omitempty"` // Output only. Max number of rows in each batch sent to the remote service. If unset, the number of rows in each batch is set dynamically.
	Remotemodelversion string `json:"remoteModelVersion,omitempty"` // Output only. The model version for LLM.
	Remoteservicetype string `json:"remoteServiceType,omitempty"` // Output only. The remote service type for remote model.
	Speechrecognizer string `json:"speechRecognizer,omitempty"` // Output only. The name of the speech recognizer to use for speech recognition. The expected format is `projects/{project}/locations/{location}/recognizers/{recognizer}`. Customers can specify this field at model creation. If not specified, a default recognizer `projects/{model project}/locations/global/recognizers/_` will be used. See more details at [recognizers](https://cloud.google.com/speech-to-text/v2/docs/reference/rest/v2/projects.locations.recognizers)
	Connection string `json:"connection,omitempty"` // Output only. Fully qualified name of the user-provided connection object of the remote model. Format: ```"projects/{project_id}/locations/{location_id}/connections/{connection_id}"```
	Endpoint string `json:"endpoint,omitempty"` // Output only. The endpoint for remote model.
}

// TableReplicationInfo represents the TableReplicationInfo schema from the OpenAPI specification
type TableReplicationInfo struct {
	Replicationstatus string `json:"replicationStatus,omitempty"` // Optional. Output only. Replication status of configured replication.
	Sourcetable TableReference `json:"sourceTable,omitempty"`
	Replicatedsourcelastrefreshtime string `json:"replicatedSourceLastRefreshTime,omitempty"` // Optional. Output only. If source is a materialized view, this field signifies the last refresh time of the source.
	Replicationerror ErrorProto `json:"replicationError,omitempty"` // Error details.
	Replicationintervalms string `json:"replicationIntervalMs,omitempty"` // Required. Specifies the interval at which the source table is polled for updates.
}

// ArimaOrder represents the ArimaOrder schema from the OpenAPI specification
type ArimaOrder struct {
	P string `json:"p,omitempty"` // Order of the autoregressive part.
	Q string `json:"q,omitempty"` // Order of the moving-average part.
	D string `json:"d,omitempty"` // Order of the differencing part.
}

// TableRow represents the TableRow schema from the OpenAPI specification
type TableRow struct {
	F []TableCell `json:"f,omitempty"` // Represents a single row in the result set, consisting of one or more fields.
}

// AuditLogConfig represents the AuditLogConfig schema from the OpenAPI specification
type AuditLogConfig struct {
	Exemptedmembers []string `json:"exemptedMembers,omitempty"` // Specifies the identities that do not cause logging for this type of permission. Follows the same format of Binding.members.
	Logtype string `json:"logType,omitempty"` // The log type that this config enables.
}

// GlobalExplanation represents the GlobalExplanation schema from the OpenAPI specification
type GlobalExplanation struct {
	Classlabel string `json:"classLabel,omitempty"` // Class label for this set of global explanations. Will be empty/null for binary logistic and linear regression models. Sorted alphabetically in descending order.
	Explanations []Explanation `json:"explanations,omitempty"` // A list of the top global explanations. Sorted by absolute value of attribution in descending order.
}

// Table represents the Table schema from the OpenAPI specification
type Table struct {
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
	Snapshotdefinition SnapshotDefinition `json:"snapshotDefinition,omitempty"` // Information about base table and snapshot time of the snapshot.
	Friendlyname string `json:"friendlyName,omitempty"` // Optional. A descriptive name for this table.
	TypeField string `json:"type,omitempty"` // Output only. Describes the table type. The following values are supported: * `TABLE`: A normal BigQuery table. * `VIEW`: A virtual table defined by a SQL query. * `EXTERNAL`: A table that references data stored in an external storage system, such as Google Cloud Storage. * `MATERIALIZED_VIEW`: A precomputed view defined by a SQL query. * `SNAPSHOT`: An immutable BigQuery table that preserves the contents of a base table at a particular time. See additional information on [table snapshots](/bigquery/docs/table-snapshots-intro). The default value is `TABLE`.
	Description string `json:"description,omitempty"` // Optional. A user-friendly description of this table.
	Tablereference TableReference `json:"tableReference,omitempty"`
	Creationtime string `json:"creationTime,omitempty"` // Output only. The time when this table was created, in milliseconds since the epoch.
	Etag string `json:"etag,omitempty"` // Output only. A hash of this resource.
	Numbytes string `json:"numBytes,omitempty"` // Output only. The size of this table in logical bytes, excluding any data in the streaming buffer.
	Maxstaleness string `json:"maxStaleness,omitempty"` // Optional. The maximum staleness of data that could be returned when the table (or stale MV) is queried. Staleness encoded as a string encoding of sql IntervalValue type.
	Numtotalphysicalbytes string `json:"numTotalPhysicalBytes,omitempty"` // Output only. The physical size of this table in bytes. This also includes storage used for time travel. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.
	Externaldataconfiguration ExternalDataConfiguration `json:"externalDataConfiguration,omitempty"`
	Materializedview MaterializedViewDefinition `json:"materializedView,omitempty"` // Definition and configuration of a materialized view.
	Expirationtime string `json:"expirationTime,omitempty"` // Optional. The time when this table expires, in milliseconds since the epoch. If not present, the table will persist indefinitely. Expired tables will be deleted and their storage reclaimed. The defaultTableExpirationMs property of the encapsulating dataset can be used to set a default expirationTime on newly created tables.
	Replicas []TableReference `json:"replicas,omitempty"` // Optional. Output only. Table references of all replicas currently active on the table.
	Kind string `json:"kind,omitempty"` // The type of resource ID.
	Numlongtermlogicalbytes string `json:"numLongTermLogicalBytes,omitempty"` // Output only. Number of logical bytes that are more than 90 days old.
	Rangepartitioning RangePartitioning `json:"rangePartitioning,omitempty"`
	Timepartitioning TimePartitioning `json:"timePartitioning,omitempty"`
	Clonedefinition CloneDefinition `json:"cloneDefinition,omitempty"` // Information about base table and clone time of a table clone.
	Materializedviewstatus MaterializedViewStatus `json:"materializedViewStatus,omitempty"` // Status of a materialized view. The last refresh timestamp status is omitted here, but is present in the MaterializedViewDefinition message.
	Numlongtermbytes string `json:"numLongTermBytes,omitempty"` // Output only. The number of logical bytes in the table that are considered "long-term storage".
	Numlongtermphysicalbytes string `json:"numLongTermPhysicalBytes,omitempty"` // Output only. Number of physical bytes more than 90 days old. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.
	Requirepartitionfilter bool `json:"requirePartitionFilter,omitempty"` // Optional. If set to true, queries over this table require a partition filter that can be used for partition elimination to be specified.
	Labels map[string]interface{} `json:"labels,omitempty"` // The labels associated with this table. You can use these to organize and group your tables. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label values are optional. Label keys must start with a letter and each label in the list must have a different key.
	Numactivephysicalbytes string `json:"numActivePhysicalBytes,omitempty"` // Output only. Number of physical bytes less than 90 days old. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.
	Clustering Clustering `json:"clustering,omitempty"` // Configures table clustering.
	Id string `json:"id,omitempty"` // Output only. An opaque ID uniquely identifying the table.
	Biglakeconfiguration BigLakeConfiguration `json:"biglakeConfiguration,omitempty"` // Configuration for BigLake managed tables.
	Numtimetravelphysicalbytes string `json:"numTimeTravelPhysicalBytes,omitempty"` // Output only. Number of physical bytes used by time travel storage (deleted or changed data). This data is not kept in real time, and might be delayed by a few seconds to a few minutes.
	Resourcetags map[string]interface{} `json:"resourceTags,omitempty"` // [Optional] The tags associated with this table. Tag keys are globally unique. See additional information on [tags](https://cloud.google.com/iam/docs/tags-access-control#definitions). An object containing a list of "key": value pairs. The key is the namespaced friendly name of the tag key, e.g. "12345/environment" where 12345 is parent id. The value is the friendly short name of the tag value, e.g. "production".
	Encryptionconfiguration EncryptionConfiguration `json:"encryptionConfiguration,omitempty"`
	Lastmodifiedtime string `json:"lastModifiedTime,omitempty"` // Output only. The time when this table was last modified, in milliseconds since the epoch.
	View ViewDefinition `json:"view,omitempty"` // Describes the definition of a logical view.
	Model ModelDefinition `json:"model,omitempty"`
	Location string `json:"location,omitempty"` // Output only. The geographic location where the table resides. This value is inherited from the dataset.
	Numtotallogicalbytes string `json:"numTotalLogicalBytes,omitempty"` // Output only. Total number of logical bytes in the table or materialized view.
	Numactivelogicalbytes string `json:"numActiveLogicalBytes,omitempty"` // Output only. Number of logical bytes that are less than 90 days old.
	Numpartitions string `json:"numPartitions,omitempty"` // Output only. The number of partitions present in the table or materialized view. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.
	Numrows string `json:"numRows,omitempty"` // Output only. The number of rows of data in this table, excluding any data in the streaming buffer.
	Tableconstraints TableConstraints `json:"tableConstraints,omitempty"` // The TableConstraints defines the primary key and foreign key.
	Streamingbuffer Streamingbuffer `json:"streamingBuffer,omitempty"`
	Defaultcollation string `json:"defaultCollation,omitempty"` // Optional. Defines the default collation specification of new STRING fields in the table. During table creation or update, if a STRING field is added to this table without explicit collation specified, then the table inherits the table default collation. A change to this field affects only fields added afterwards, and does not alter the existing fields. The following values are supported: * 'und:ci': undetermined locale, case insensitive. * '': empty string. Default to case-sensitive behavior.
	Tablereplicationinfo TableReplicationInfo `json:"tableReplicationInfo,omitempty"` // Replication info of a table created using `AS REPLICA` DDL like: `CREATE MATERIALIZED VIEW mv1 AS REPLICA OF src_mv`
	Numphysicalbytes string `json:"numPhysicalBytes,omitempty"` // Output only. The physical size of this table in bytes. This includes storage used for time travel.
	Selflink string `json:"selfLink,omitempty"` // Output only. A URL that can be used to access this resource again.
	Defaultroundingmode string `json:"defaultRoundingMode,omitempty"` // Optional. Defines the default rounding mode specification of new decimal fields (NUMERIC OR BIGNUMERIC) in the table. During table creation or update, if a decimal field is added to this table without an explicit rounding mode specified, then the field inherits the table default rounding mode. Changing this field doesn't affect existing fields.
}

// DoubleRange represents the DoubleRange schema from the OpenAPI specification
type DoubleRange struct {
	Max float64 `json:"max,omitempty"` // Max value of the double parameter.
	Min float64 `json:"min,omitempty"` // Min value of the double parameter.
}

// Job represents the Job schema from the OpenAPI specification
type Job struct {
	Id string `json:"id,omitempty"` // Output only. Opaque ID field of the job.
	Jobcreationreason JobCreationReason `json:"jobCreationReason,omitempty"` // Reason about why a Job was created from a [`jobs.query`](https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query) method when used with `JOB_CREATION_OPTIONAL` Job creation mode. For [`jobs.insert`](https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert) method calls it will always be `REQUESTED`. This feature is not yet available. Jobs will always be created.
	Configuration JobConfiguration `json:"configuration,omitempty"`
	Selflink string `json:"selfLink,omitempty"` // Output only. A URL that can be used to access the resource again.
	Status JobStatus `json:"status,omitempty"`
	User_email string `json:"user_email,omitempty"` // Output only. Email address of the user who ran the job.
	Jobreference JobReference `json:"jobReference,omitempty"` // A job reference is a fully qualified identifier for referring to a job.
	Principal_subject string `json:"principal_subject,omitempty"` // Output only. [Full-projection-only] String representation of identity of requesting party. Populated for both first- and third-party identities. Only present for APIs that support third-party identities.
	Kind string `json:"kind,omitempty"` // Output only. The type of the resource.
	Statistics JobStatistics `json:"statistics,omitempty"` // Statistics for a single job execution.
	Etag string `json:"etag,omitempty"` // Output only. A hash of this resource.
}

// StringHparamSearchSpace represents the StringHparamSearchSpace schema from the OpenAPI specification
type StringHparamSearchSpace struct {
	Candidates []string `json:"candidates,omitempty"` // Canididates for the string or enum parameter in lower case.
}

// TableDataInsertAllRequest represents the TableDataInsertAllRequest schema from the OpenAPI specification
type TableDataInsertAllRequest struct {
	Skipinvalidrows bool `json:"skipInvalidRows,omitempty"` // Optional. Insert all valid rows of a request, even if invalid rows exist. The default value is false, which causes the entire request to fail if any invalid rows exist.
	Templatesuffix string `json:"templateSuffix,omitempty"` // Optional. If specified, treats the destination table as a base template, and inserts the rows into an instance table named "{destination}{templateSuffix}". BigQuery will manage creation of the instance table, using the schema of the base template table. See https://cloud.google.com/bigquery/streaming-data-into-bigquery#template-tables for considerations when working with templates tables.
	Traceid string `json:"traceId,omitempty"` // Optional. Unique request trace id. Used for debugging purposes only. It is case-sensitive, limited to up to 36 ASCII characters. A UUID is recommended.
	Ignoreunknownvalues bool `json:"ignoreUnknownValues,omitempty"` // Optional. Accept rows that contain values that do not match the schema. The unknown values are ignored. Default is false, which treats unknown values as errors.
	Kind string `json:"kind,omitempty"` // Optional. The resource type of the response. The value is not checked at the backend. Historically, it has been set to "bigquery#tableDataInsertAllRequest" but you are not required to set it.
	Rows []map[string]interface{} `json:"rows,omitempty"`
}

// TransactionInfo represents the TransactionInfo schema from the OpenAPI specification
type TransactionInfo struct {
	Transactionid string `json:"transactionId,omitempty"` // Output only. [Alpha] Id of the transaction.
}

// SparkOptions represents the SparkOptions schema from the OpenAPI specification
type SparkOptions struct {
	Fileuris []string `json:"fileUris,omitempty"` // Files to be placed in the working directory of each executor. For more information about Apache Spark, see [Apache Spark](https://spark.apache.org/docs/latest/index.html).
	Properties map[string]interface{} `json:"properties,omitempty"` // Configuration properties as a set of key/value pairs, which will be passed on to the Spark application. For more information, see [Apache Spark](https://spark.apache.org/docs/latest/index.html) and the [procedure option list](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#procedure_option_list).
	Archiveuris []string `json:"archiveUris,omitempty"` // Archive files to be extracted into the working directory of each executor. For more information about Apache Spark, see [Apache Spark](https://spark.apache.org/docs/latest/index.html).
	Connection string `json:"connection,omitempty"` // Fully qualified name of the user-provided Spark connection object. Format: ```"projects/{project_id}/locations/{location_id}/connections/{connection_id}"```
	Mainclass string `json:"mainClass,omitempty"` // The fully qualified name of a class in jar_uris, for example, com.example.wordcount. Exactly one of main_class and main_jar_uri field should be set for Java/Scala language type.
	Mainfileuri string `json:"mainFileUri,omitempty"` // The main file/jar URI of the Spark application. Exactly one of the definition_body field and the main_file_uri field must be set for Python. Exactly one of main_class and main_file_uri field should be set for Java/Scala language type.
	Pyfileuris []string `json:"pyFileUris,omitempty"` // Python files to be placed on the PYTHONPATH for PySpark application. Supported file types: `.py`, `.egg`, and `.zip`. For more information about Apache Spark, see [Apache Spark](https://spark.apache.org/docs/latest/index.html).
	Jaruris []string `json:"jarUris,omitempty"` // JARs to include on the driver and executor CLASSPATH. For more information about Apache Spark, see [Apache Spark](https://spark.apache.org/docs/latest/index.html).
	Runtimeversion string `json:"runtimeVersion,omitempty"` // Runtime version. If not specified, the default runtime version is used.
	Containerimage string `json:"containerImage,omitempty"` // Custom container image for the runtime environment.
}

// QueryParameterValue represents the QueryParameterValue schema from the OpenAPI specification
type QueryParameterValue struct {
	Structvalues map[string]interface{} `json:"structValues,omitempty"` // The struct field values.
	Value string `json:"value,omitempty"` // Optional. The value of this value, if a simple scalar type.
	Arrayvalues []QueryParameterValue `json:"arrayValues,omitempty"` // Optional. The array values, if this is an array type.
	Rangevalue RangeValue `json:"rangeValue,omitempty"` // Represents the value of a range.
}

// Clustering represents the Clustering schema from the OpenAPI specification
type Clustering struct {
	Fields []string `json:"fields,omitempty"` // One or more fields on which data should be clustered. Only top-level, non-repeated, simple-type fields are supported. The ordering of the clustering fields should be prioritized from most to least important for filtering purposes. Additional information on limitations can be found here: https://cloud.google.com/bigquery/docs/creating-clustered-tables#limitations
}

// TableMetadataCacheUsage represents the TableMetadataCacheUsage schema from the OpenAPI specification
type TableMetadataCacheUsage struct {
	Explanation string `json:"explanation,omitempty"` // Free form human-readable reason metadata caching was unused for the job.
	Tablereference TableReference `json:"tableReference,omitempty"`
	Tabletype string `json:"tableType,omitempty"` // [Table type](/bigquery/docs/reference/rest/v2/tables#Table.FIELDS.type).
	Unusedreason string `json:"unusedReason,omitempty"` // Reason for not using metadata caching for the table.
}

// ArimaCoefficients represents the ArimaCoefficients schema from the OpenAPI specification
type ArimaCoefficients struct {
	Autoregressivecoefficients []float64 `json:"autoRegressiveCoefficients,omitempty"` // Auto-regressive coefficients, an array of double.
	Interceptcoefficient float64 `json:"interceptCoefficient,omitempty"` // Intercept coefficient, just a double not an array.
	Movingaveragecoefficients []float64 `json:"movingAverageCoefficients,omitempty"` // Moving-average coefficients, an array of double.
}

// SnapshotDefinition represents the SnapshotDefinition schema from the OpenAPI specification
type SnapshotDefinition struct {
	Snapshottime string `json:"snapshotTime,omitempty"` // Required. The time at which the base table was snapshot. This value is reported in the JSON response using RFC3339 format.
	Basetablereference TableReference `json:"baseTableReference,omitempty"`
}

// JsonObject represents the JsonObject schema from the OpenAPI specification
type JsonObject struct {
}

// IntCandidates represents the IntCandidates schema from the OpenAPI specification
type IntCandidates struct {
	Candidates []string `json:"candidates,omitempty"` // Candidates for the int parameter in increasing order.
}

// PerformanceInsights represents the PerformanceInsights schema from the OpenAPI specification
type PerformanceInsights struct {
	Stageperformancechangeinsights []StagePerformanceChangeInsight `json:"stagePerformanceChangeInsights,omitempty"` // Output only. Query stage performance insights compared to previous runs, for diagnosing performance regression.
	Stageperformancestandaloneinsights []StagePerformanceStandaloneInsight `json:"stagePerformanceStandaloneInsights,omitempty"` // Output only. Standalone query stage performance insights, for exploring potential improvements.
	Avgpreviousexecutionms string `json:"avgPreviousExecutionMs,omitempty"` // Output only. Average execution ms of previous runs. Indicates the job ran slow compared to previous executions. To find previous executions, use INFORMATION_SCHEMA tables and filter jobs with same query hash.
}

// VectorSearchStatistics represents the VectorSearchStatistics schema from the OpenAPI specification
type VectorSearchStatistics struct {
	Indexusagemode string `json:"indexUsageMode,omitempty"` // Specifies the index usage mode for the query.
	Indexunusedreasons []IndexUnusedReason `json:"indexUnusedReasons,omitempty"` // When `indexUsageMode` is `UNUSED` or `PARTIALLY_USED`, this field explains why indexes were not used in all or part of the vector search query. If `indexUsageMode` is `FULLY_USED`, this field is not populated.
}

// StandardSqlField represents the StandardSqlField schema from the OpenAPI specification
type StandardSqlField struct {
	Name string `json:"name,omitempty"` // Optional. The name of this field. Can be absent for struct fields.
	TypeField StandardSqlDataType `json:"type,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
}

// ListModelsResponse represents the ListModelsResponse schema from the OpenAPI specification
type ListModelsResponse struct {
	Models []Model `json:"models,omitempty"` // Models in the requested dataset. Only the following fields are populated: model_reference, model_type, creation_time, last_modified_time and labels.
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token to request the next page of results.
}

// ExportDataStatistics represents the ExportDataStatistics schema from the OpenAPI specification
type ExportDataStatistics struct {
	Rowcount string `json:"rowCount,omitempty"` // [Alpha] Number of destination rows generated in case of EXPORT DATA statement only.
	Filecount string `json:"fileCount,omitempty"` // Number of destination files generated in case of EXPORT DATA statement only.
}

// ViewDefinition represents the ViewDefinition schema from the OpenAPI specification
type ViewDefinition struct {
	Userdefinedfunctionresources []UserDefinedFunctionResource `json:"userDefinedFunctionResources,omitempty"` // Describes user-defined function resources used in the query.
	Privacypolicy PrivacyPolicy `json:"privacyPolicy,omitempty"` // Represents privacy policy that contains the privacy requirements specified by the data owner. Currently, this is only supported on views.
	Query string `json:"query,omitempty"` // Required. A query that BigQuery executes when the view is referenced.
	Useexplicitcolumnnames bool `json:"useExplicitColumnNames,omitempty"` // True if the column names are explicitly specified. For example by using the 'CREATE VIEW v(c1, c2) AS ...' syntax. Can only be set for GoogleSQL views.
	Uselegacysql bool `json:"useLegacySql,omitempty"` // Specifies whether to use BigQuery's legacy SQL for this view. The default value is true. If set to false, the view will use BigQuery's GoogleSQL: https://cloud.google.com/bigquery/sql-reference/ Queries and views that reference this view must use the same flag value. A wrapper is used here because the default value is True.
}

// Expr represents the Expr schema from the OpenAPI specification
type Expr struct {
	Location string `json:"location,omitempty"` // Optional. String indicating the location of the expression for error reporting, e.g. a file name and a position in the file.
	Title string `json:"title,omitempty"` // Optional. Title for the expression, i.e. a short string describing its purpose. This can be used e.g. in UIs which allow to enter the expression.
	Description string `json:"description,omitempty"` // Optional. Description of the expression. This is a longer text which describes the expression, e.g. when hovered over it in a UI.
	Expression string `json:"expression,omitempty"` // Textual representation of an expression in Common Expression Language syntax.
}

// ListRowAccessPoliciesResponse represents the ListRowAccessPoliciesResponse schema from the OpenAPI specification
type ListRowAccessPoliciesResponse struct {
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token to request the next page of results.
	Rowaccesspolicies []RowAccessPolicy `json:"rowAccessPolicies,omitempty"` // Row access policies on the requested table.
}

// CloneDefinition represents the CloneDefinition schema from the OpenAPI specification
type CloneDefinition struct {
	Basetablereference TableReference `json:"baseTableReference,omitempty"`
	Clonetime string `json:"cloneTime,omitempty"` // Required. The time at which the base table was cloned. This value is reported in the JSON response using RFC3339 format.
}

// RangeValue represents the RangeValue schema from the OpenAPI specification
type RangeValue struct {
	Start QueryParameterValue `json:"start,omitempty"` // The value of a query parameter.
	End QueryParameterValue `json:"end,omitempty"` // The value of a query parameter.
}

// DestinationTableProperties represents the DestinationTableProperties schema from the OpenAPI specification
type DestinationTableProperties struct {
	Description string `json:"description,omitempty"` // Optional. The description for the destination table. This will only be used if the destination table is newly created. If the table already exists and a value different than the current description is provided, the job will fail.
	Expirationtime string `json:"expirationTime,omitempty"` // Internal use only.
	Friendlyname string `json:"friendlyName,omitempty"` // Optional. Friendly name for the destination table. If the table already exists, it should be same as the existing friendly name.
	Labels map[string]interface{} `json:"labels,omitempty"` // Optional. The labels associated with this table. You can use these to organize and group your tables. This will only be used if the destination table is newly created. If the table already exists and labels are different than the current labels are provided, the job will fail.
}

// ExternalDataConfiguration represents the ExternalDataConfiguration schema from the OpenAPI specification
type ExternalDataConfiguration struct {
	Googlesheetsoptions GoogleSheetsOptions `json:"googleSheetsOptions,omitempty"` // Options specific to Google Sheets data sources.
	Ignoreunknownvalues bool `json:"ignoreUnknownValues,omitempty"` // Optional. Indicates if BigQuery should allow extra values that are not represented in the table schema. If true, the extra values are ignored. If false, records with extra columns are treated as bad records, and if there are too many bad records, an invalid error is returned in the job result. The default value is false. The sourceFormat property determines what BigQuery treats as an extra value: CSV: Trailing columns JSON: Named values that don't match any column names Google Cloud Bigtable: This setting is ignored. Google Cloud Datastore backups: This setting is ignored. Avro: This setting is ignored. ORC: This setting is ignored. Parquet: This setting is ignored.
	Objectmetadata string `json:"objectMetadata,omitempty"` // Optional. ObjectMetadata is used to create Object Tables. Object Tables contain a listing of objects (with their metadata) found at the source_uris. If ObjectMetadata is set, source_format should be omitted. Currently SIMPLE is the only supported Object Metadata type.
	Decimaltargettypes []string `json:"decimalTargetTypes,omitempty"` // Defines the list of possible SQL data types to which the source decimal values are converted. This list and the precision and the scale parameters of the decimal field determine the target type. In the order of NUMERIC, BIGNUMERIC, and STRING, a type is picked if it is in the specified list and if it supports the precision and the scale. STRING supports all precision and scale values. If none of the listed types supports the precision and the scale, the type supporting the widest range in the specified list is picked, and if a value exceeds the supported range when reading the data, an error will be thrown. Example: Suppose the value of this field is ["NUMERIC", "BIGNUMERIC"]. If (precision,scale) is: * (38,9) -> NUMERIC; * (39,9) -> BIGNUMERIC (NUMERIC cannot hold 30 integer digits); * (38,10) -> BIGNUMERIC (NUMERIC cannot hold 10 fractional digits); * (76,38) -> BIGNUMERIC; * (77,38) -> BIGNUMERIC (error if value exeeds supported range). This field cannot contain duplicate types. The order of the types in this field is ignored. For example, ["BIGNUMERIC", "NUMERIC"] is the same as ["NUMERIC", "BIGNUMERIC"] and NUMERIC always takes precedence over BIGNUMERIC. Defaults to ["NUMERIC", "STRING"] for ORC and ["NUMERIC"] for the other file formats.
	Hivepartitioningoptions HivePartitioningOptions `json:"hivePartitioningOptions,omitempty"` // Options for configuring hive partitioning detect.
	Jsonextension string `json:"jsonExtension,omitempty"` // Optional. Load option to be used together with source_format newline-delimited JSON to indicate that a variant of JSON is being loaded. To load newline-delimited GeoJSON, specify GEOJSON (and source_format must be set to NEWLINE_DELIMITED_JSON).
	Compression string `json:"compression,omitempty"` // Optional. The compression type of the data source. Possible values include GZIP and NONE. The default value is NONE. This setting is ignored for Google Cloud Bigtable, Google Cloud Datastore backups, Avro, ORC and Parquet formats. An empty string is an invalid value.
	Autodetect bool `json:"autodetect,omitempty"` // Try to detect schema and format options automatically. Any option specified explicitly will be honored.
	Avrooptions AvroOptions `json:"avroOptions,omitempty"` // Options for external data sources.
	Sourceformat string `json:"sourceFormat,omitempty"` // [Required] The data format. For CSV files, specify "CSV". For Google sheets, specify "GOOGLE_SHEETS". For newline-delimited JSON, specify "NEWLINE_DELIMITED_JSON". For Avro files, specify "AVRO". For Google Cloud Datastore backups, specify "DATASTORE_BACKUP". For Apache Iceberg tables, specify "ICEBERG". For ORC files, specify "ORC". For Parquet files, specify "PARQUET". [Beta] For Google Cloud Bigtable, specify "BIGTABLE".
	Bigtableoptions BigtableOptions `json:"bigtableOptions,omitempty"` // Options specific to Google Cloud Bigtable data sources.
	Connectionid string `json:"connectionId,omitempty"` // Optional. The connection specifying the credentials to be used to read external storage, such as Azure Blob, Cloud Storage, or S3. The connection_id can have the form "<project\_id>.<location\_id>.<connection\_id>" or "projects/<project\_id>/locations/<location\_id>/connections/<connection\_id>".
	Metadatacachemode string `json:"metadataCacheMode,omitempty"` // Optional. Metadata Cache Mode for the table. Set this to enable caching of metadata from external data source.
	Filesetspectype string `json:"fileSetSpecType,omitempty"` // Optional. Specifies how source URIs are interpreted for constructing the file set to load. By default source URIs are expanded against the underlying storage. Other options include specifying manifest files. Only applicable to object storage systems.
	Jsonoptions JsonOptions `json:"jsonOptions,omitempty"` // Json Options for load and make external tables.
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
	Sourceuris []string `json:"sourceUris,omitempty"` // [Required] The fully-qualified URIs that point to your data in Google Cloud. For Google Cloud Storage URIs: Each URI can contain one '*' wildcard character and it must come after the 'bucket' name. Size limits related to load jobs apply to external data sources. For Google Cloud Bigtable URIs: Exactly one URI can be specified and it has be a fully specified and valid HTTPS URL for a Google Cloud Bigtable table. For Google Cloud Datastore backups, exactly one URI can be specified. Also, the '*' wildcard character is not allowed.
	Referencefileschemauri string `json:"referenceFileSchemaUri,omitempty"` // Optional. When creating an external table, the user can provide a reference file with the table schema. This is enabled for the following formats: AVRO, PARQUET, ORC.
	Csvoptions CsvOptions `json:"csvOptions,omitempty"` // Information related to a CSV data source.
	Maxbadrecords int `json:"maxBadRecords,omitempty"` // Optional. The maximum number of bad records that BigQuery can ignore when reading data. If the number of bad records exceeds this value, an invalid error is returned in the job result. The default value is 0, which requires that all records are valid. This setting is ignored for Google Cloud Bigtable, Google Cloud Datastore backups, Avro, ORC and Parquet formats.
	Parquetoptions ParquetOptions `json:"parquetOptions,omitempty"` // Parquet Options for load and make external tables.
}

// ScriptStackFrame represents the ScriptStackFrame schema from the OpenAPI specification
type ScriptStackFrame struct {
	Startcolumn int `json:"startColumn,omitempty"` // Output only. One-based start column.
	Startline int `json:"startLine,omitempty"` // Output only. One-based start line.
	Text string `json:"text,omitempty"` // Output only. Text of the current statement/expression.
	Endcolumn int `json:"endColumn,omitempty"` // Output only. One-based end column.
	Endline int `json:"endLine,omitempty"` // Output only. One-based end line.
	Procedureid string `json:"procedureId,omitempty"` // Output only. Name of the active procedure, empty if in a top-level script.
}

// QueryRequest represents the QueryRequest schema from the OpenAPI specification
type QueryRequest struct {
	Createsession bool `json:"createSession,omitempty"` // Optional. If true, creates a new session using a randomly generated session_id. If false, runs query with an existing session_id passed in ConnectionProperty, otherwise runs query in non-session mode. The session location will be set to QueryRequest.location if it is present, otherwise it's set to the default location based on existing routing logic.
	Maxresults int `json:"maxResults,omitempty"` // Optional. The maximum number of rows of data to return per page of results. Setting this flag to a small value such as 1000 and then paging through results might improve reliability when the query result set is large. In addition to this limit, responses are also limited to 10 MB. By default, there is no maximum row count, and only the byte limit applies.
	Timeoutms int `json:"timeoutMs,omitempty"` // Optional. Optional: Specifies the maximum amount of time, in milliseconds, that the client is willing to wait for the query to complete. By default, this limit is 10 seconds (10,000 milliseconds). If the query is complete, the jobComplete field in the response is true. If the query has not yet completed, jobComplete is false. You can request a longer timeout period in the timeoutMs field. However, the call is not guaranteed to wait for the specified timeout; it typically returns after around 200 seconds (200,000 milliseconds), even if the query is not complete. If jobComplete is false, you can continue to wait for the query to complete by calling the getQueryResults method until the jobComplete field in the getQueryResults response is true.
	Formatoptions DataFormatOptions `json:"formatOptions,omitempty"` // Options for data format adjustments.
	Requestid string `json:"requestId,omitempty"` // Optional. A unique user provided identifier to ensure idempotent behavior for queries. Note that this is different from the job_id. It has the following properties: 1. It is case-sensitive, limited to up to 36 ASCII characters. A UUID is recommended. 2. Read only queries can ignore this token since they are nullipotent by definition. 3. For the purposes of idempotency ensured by the request_id, a request is considered duplicate of another only if they have the same request_id and are actually duplicates. When determining whether a request is a duplicate of another request, all parameters in the request that may affect the result are considered. For example, query, connection_properties, query_parameters, use_legacy_sql are parameters that affect the result and are considered when determining whether a request is a duplicate, but properties like timeout_ms don't affect the result and are thus not considered. Dry run query requests are never considered duplicate of another request. 4. When a duplicate mutating query request is detected, it returns: a. the results of the mutation if it completes successfully within the timeout. b. the running operation if it is still in progress at the end of the timeout. 5. Its lifetime is limited to 15 minutes. In other words, if two requests are sent with the same request_id, but more than 15 minutes apart, idempotency is not guaranteed.
	Labels map[string]interface{} `json:"labels,omitempty"` // Optional. The labels associated with this query. Labels can be used to organize and group query jobs. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label keys must start with a letter and each label in the list must have a different key.
	Continuous bool `json:"continuous,omitempty"` // [Optional] Specifies whether the query should be executed as a continuous query. The default value is false.
	Uselegacysql bool `json:"useLegacySql,omitempty"` // Specifies whether to use BigQuery's legacy SQL dialect for this query. The default value is true. If set to false, the query will use BigQuery's GoogleSQL: https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is set to false, the value of flattenResults is ignored; query will be run as if flattenResults is false.
	Preservenulls bool `json:"preserveNulls,omitempty"` // This property is deprecated.
	Query string `json:"query,omitempty"` // Required. A query string to execute, using Google Standard SQL or legacy SQL syntax. Example: "SELECT COUNT(f1) FROM myProjectId.myDatasetId.myTableId".
	Kind string `json:"kind,omitempty"` // The resource type of the request.
	Maximumbytesbilled string `json:"maximumBytesBilled,omitempty"` // Optional. Limits the bytes billed for this query. Queries with bytes billed above this limit will fail (without incurring a charge). If unspecified, the project default is used.
	Jobcreationmode string `json:"jobCreationMode,omitempty"` // Optional. If not set, jobs are always required. If set, the query request will follow the behavior described JobCreationMode. This feature is not yet available. Jobs will always be created.
	Queryparameters []QueryParameter `json:"queryParameters,omitempty"` // Query parameters for GoogleSQL queries.
	Dryrun bool `json:"dryRun,omitempty"` // Optional. If set to true, BigQuery doesn't run the job. Instead, if the query is valid, BigQuery returns statistics about the job such as how many bytes would be processed. If the query is invalid, an error returns. The default value is false.
	Usequerycache bool `json:"useQueryCache,omitempty"` // Optional. Whether to look for the result in the query cache. The query cache is a best-effort cache that will be flushed whenever tables in the query are modified. The default value is true.
	Defaultdataset DatasetReference `json:"defaultDataset,omitempty"`
	Location string `json:"location,omitempty"` // The geographic location where the job should run. See details at https://cloud.google.com/bigquery/docs/locations#specifying_your_location.
	Parametermode string `json:"parameterMode,omitempty"` // GoogleSQL only. Set to POSITIONAL to use positional (?) query parameters or to NAMED to use named (@myparam) query parameters in this query.
	Connectionproperties []ConnectionProperty `json:"connectionProperties,omitempty"` // Optional. Connection properties which can modify the query behavior.
}

// MaterializedViewStatistics represents the MaterializedViewStatistics schema from the OpenAPI specification
type MaterializedViewStatistics struct {
	Materializedview []MaterializedView `json:"materializedView,omitempty"` // Materialized views considered for the query job. Only certain materialized views are used. For a detailed list, see the child message. If many materialized views are considered, then the list might be incomplete.
}

// MetadataCacheStatistics represents the MetadataCacheStatistics schema from the OpenAPI specification
type MetadataCacheStatistics struct {
	Tablemetadatacacheusage []TableMetadataCacheUsage `json:"tableMetadataCacheUsage,omitempty"` // Set for the Metadata caching eligible tables referenced in the query.
}

// IterationResult represents the IterationResult schema from the OpenAPI specification
type IterationResult struct {
	Principalcomponentinfos []PrincipalComponentInfo `json:"principalComponentInfos,omitempty"` // The information of the principal components.
	Trainingloss float64 `json:"trainingLoss,omitempty"` // Loss computed on the training data at the end of iteration.
	Arimaresult ArimaResult `json:"arimaResult,omitempty"` // (Auto-)arima fitting result. Wrap everything in ArimaResult for easier refactoring if we want to use model-specific iteration results.
	Clusterinfos []ClusterInfo `json:"clusterInfos,omitempty"` // Information about top clusters for clustering models.
	Durationms string `json:"durationMs,omitempty"` // Time taken to run the iteration in milliseconds.
	Evalloss float64 `json:"evalLoss,omitempty"` // Loss computed on the eval data at the end of iteration.
	Index int `json:"index,omitempty"` // Index of the iteration, 0 based.
	Learnrate float64 `json:"learnRate,omitempty"` // Learn rate used for this iteration.
}

// JobConfigurationExtract represents the JobConfigurationExtract schema from the OpenAPI specification
type JobConfigurationExtract struct {
	Destinationuri string `json:"destinationUri,omitempty"` // [Pick one] DEPRECATED: Use destinationUris instead, passing only one URI as necessary. The fully-qualified Google Cloud Storage URI where the extracted table should be written.
	Destinationuris []string `json:"destinationUris,omitempty"` // [Pick one] A list of fully-qualified Google Cloud Storage URIs where the extracted table should be written.
	Fielddelimiter string `json:"fieldDelimiter,omitempty"` // Optional. When extracting data in CSV format, this defines the delimiter to use between fields in the exported data. Default is ','. Not applicable when extracting models.
	Useavrologicaltypes bool `json:"useAvroLogicalTypes,omitempty"` // Whether to use logical types when extracting to AVRO format. Not applicable when extracting models.
	Compression string `json:"compression,omitempty"` // Optional. The compression type to use for exported files. Possible values include DEFLATE, GZIP, NONE, SNAPPY, and ZSTD. The default value is NONE. Not all compression formats are support for all file formats. DEFLATE is only supported for Avro. ZSTD is only supported for Parquet. Not applicable when extracting models.
	Destinationformat string `json:"destinationFormat,omitempty"` // Optional. The exported file format. Possible values include CSV, NEWLINE_DELIMITED_JSON, PARQUET, or AVRO for tables and ML_TF_SAVED_MODEL or ML_XGBOOST_BOOSTER for models. The default value for tables is CSV. Tables with nested or repeated fields cannot be exported as CSV. The default value for models is ML_TF_SAVED_MODEL.
	Printheader bool `json:"printHeader,omitempty"` // Optional. Whether to print out a header row in the results. Default is true. Not applicable when extracting models.
	Modelextractoptions ModelExtractOptions `json:"modelExtractOptions,omitempty"` // Options related to model extraction.
	Sourcemodel ModelReference `json:"sourceModel,omitempty"` // Id path of a model.
	Sourcetable TableReference `json:"sourceTable,omitempty"`
}

// JobStatistics4 represents the JobStatistics4 schema from the OpenAPI specification
type JobStatistics4 struct {
	Destinationurifilecounts []string `json:"destinationUriFileCounts,omitempty"` // Output only. Number of files per destination URI or URI pattern specified in the extract configuration. These values will be in the same order as the URIs specified in the 'destinationUris' field.
	Inputbytes string `json:"inputBytes,omitempty"` // Output only. Number of user bytes extracted into the result. This is the byte count as computed by BigQuery for billing purposes and doesn't have any relationship with the number of actual result bytes extracted in the desired format.
	Timeline []QueryTimelineSample `json:"timeline,omitempty"` // Output only. Describes a timeline of job execution.
}

// JsonOptions represents the JsonOptions schema from the OpenAPI specification
type JsonOptions struct {
	Encoding string `json:"encoding,omitempty"` // Optional. The character encoding of the data. The supported values are UTF-8, UTF-16BE, UTF-16LE, UTF-32BE, and UTF-32LE. The default value is UTF-8.
}

// StagePerformanceChangeInsight represents the StagePerformanceChangeInsight schema from the OpenAPI specification
type StagePerformanceChangeInsight struct {
	Inputdatachange InputDataChange `json:"inputDataChange,omitempty"` // Details about the input data change insight.
	Stageid string `json:"stageId,omitempty"` // Output only. The stage id that the insight mapped to.
}

// JobConfigurationLoad represents the JobConfigurationLoad schema from the OpenAPI specification
type JobConfigurationLoad struct {
	Parquetoptions ParquetOptions `json:"parquetOptions,omitempty"` // Parquet Options for load and make external tables.
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
	Destinationencryptionconfiguration EncryptionConfiguration `json:"destinationEncryptionConfiguration,omitempty"`
	Preserveasciicontrolcharacters bool `json:"preserveAsciiControlCharacters,omitempty"` // Optional. When sourceFormat is set to "CSV", this indicates whether the embedded ASCII control characters (the first 32 characters in the ASCII-table, from '\x00' to '\x1F') are preserved.
	Createsession bool `json:"createSession,omitempty"` // Optional. If this property is true, the job creates a new session using a randomly generated session_id. To continue using a created session with subsequent queries, pass the existing session identifier as a `ConnectionProperty` value. The session identifier is returned as part of the `SessionInfo` message within the query statistics. The new session's location will be set to `Job.JobReference.location` if it is present, otherwise it's set to the default location based on existing routing logic.
	Referencefileschemauri string `json:"referenceFileSchemaUri,omitempty"` // Optional. The user can provide a reference file with the reader schema. This file is only loaded if it is part of source URIs, but is not loaded otherwise. It is enabled for the following formats: AVRO, PARQUET, ORC.
	Sourceformat string `json:"sourceFormat,omitempty"` // Optional. The format of the data files. For CSV files, specify "CSV". For datastore backups, specify "DATASTORE_BACKUP". For newline-delimited JSON, specify "NEWLINE_DELIMITED_JSON". For Avro, specify "AVRO". For parquet, specify "PARQUET". For orc, specify "ORC". The default value is CSV.
	Sourceuris []string `json:"sourceUris,omitempty"` // [Required] The fully-qualified URIs that point to your data in Google Cloud. For Google Cloud Storage URIs: Each URI can contain one '*' wildcard character and it must come after the 'bucket' name. Size limits related to load jobs apply to external data sources. For Google Cloud Bigtable URIs: Exactly one URI can be specified and it has be a fully specified and valid HTTPS URL for a Google Cloud Bigtable table. For Google Cloud Datastore backups: Exactly one URI can be specified. Also, the '*' wildcard character is not allowed.
	Allowjaggedrows bool `json:"allowJaggedRows,omitempty"` // Optional. Accept rows that are missing trailing optional columns. The missing values are treated as nulls. If false, records with missing trailing columns are treated as bad records, and if there are too many bad records, an invalid error is returned in the job result. The default value is false. Only applicable to CSV, ignored for other formats.
	Connectionproperties []ConnectionProperty `json:"connectionProperties,omitempty"` // Optional. Connection properties which can modify the load job behavior. Currently, only the 'session_id' connection property is supported, and is used to resolve _SESSION appearing as the dataset id.
	Clustering Clustering `json:"clustering,omitempty"` // Configures table clustering.
	Decimaltargettypes []string `json:"decimalTargetTypes,omitempty"` // Defines the list of possible SQL data types to which the source decimal values are converted. This list and the precision and the scale parameters of the decimal field determine the target type. In the order of NUMERIC, BIGNUMERIC, and STRING, a type is picked if it is in the specified list and if it supports the precision and the scale. STRING supports all precision and scale values. If none of the listed types supports the precision and the scale, the type supporting the widest range in the specified list is picked, and if a value exceeds the supported range when reading the data, an error will be thrown. Example: Suppose the value of this field is ["NUMERIC", "BIGNUMERIC"]. If (precision,scale) is: * (38,9) -> NUMERIC; * (39,9) -> BIGNUMERIC (NUMERIC cannot hold 30 integer digits); * (38,10) -> BIGNUMERIC (NUMERIC cannot hold 10 fractional digits); * (76,38) -> BIGNUMERIC; * (77,38) -> BIGNUMERIC (error if value exeeds supported range). This field cannot contain duplicate types. The order of the types in this field is ignored. For example, ["BIGNUMERIC", "NUMERIC"] is the same as ["NUMERIC", "BIGNUMERIC"] and NUMERIC always takes precedence over BIGNUMERIC. Defaults to ["NUMERIC", "STRING"] for ORC and ["NUMERIC"] for the other file formats.
	Maxbadrecords int `json:"maxBadRecords,omitempty"` // Optional. The maximum number of bad records that BigQuery can ignore when running the job. If the number of bad records exceeds this value, an invalid error is returned in the job result. The default value is 0, which requires that all records are valid. This is only supported for CSV and NEWLINE_DELIMITED_JSON file formats.
	Timepartitioning TimePartitioning `json:"timePartitioning,omitempty"`
	Encoding string `json:"encoding,omitempty"` // Optional. The character encoding of the data. The supported values are UTF-8, ISO-8859-1, UTF-16BE, UTF-16LE, UTF-32BE, and UTF-32LE. The default value is UTF-8. BigQuery decodes the data after the raw, binary data has been split using the values of the `quote` and `fieldDelimiter` properties. If you don't specify an encoding, or if you specify a UTF-8 encoding when the CSV file is not UTF-8 encoded, BigQuery attempts to convert the data to UTF-8. Generally, your data loads successfully, but it may not match byte-for-byte what you expect. To avoid this, specify the correct encoding by using the `--encoding` flag. If BigQuery can't convert a character other than the ASCII `0` character, BigQuery converts the character to the standard Unicode replacement character: �.
	Rangepartitioning RangePartitioning `json:"rangePartitioning,omitempty"`
	Writedisposition string `json:"writeDisposition,omitempty"` // Optional. Specifies the action that occurs if the destination table already exists. The following values are supported: * WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the data, removes the constraints and uses the schema from the load job. * WRITE_APPEND: If the table already exists, BigQuery appends the data to the table. * WRITE_EMPTY: If the table already exists and contains data, a 'duplicate' error is returned in the job result. The default value is WRITE_APPEND. Each action is atomic and only occurs if BigQuery is able to complete the job successfully. Creation, truncation and append actions occur as one atomic update upon job completion.
	Ignoreunknownvalues bool `json:"ignoreUnknownValues,omitempty"` // Optional. Indicates if BigQuery should allow extra values that are not represented in the table schema. If true, the extra values are ignored. If false, records with extra columns are treated as bad records, and if there are too many bad records, an invalid error is returned in the job result. The default value is false. The sourceFormat property determines what BigQuery treats as an extra value: CSV: Trailing columns JSON: Named values that don't match any column names in the table schema Avro, Parquet, ORC: Fields in the file schema that don't exist in the table schema.
	Jsonextension string `json:"jsonExtension,omitempty"` // Optional. Load option to be used together with source_format newline-delimited JSON to indicate that a variant of JSON is being loaded. To load newline-delimited GeoJSON, specify GEOJSON (and source_format must be set to NEWLINE_DELIMITED_JSON).
	Copyfilesonly bool `json:"copyFilesOnly,omitempty"` // Optional. [Experimental] Configures the load job to only copy files to the destination BigLake managed table with an external storage_uri, without reading file content and writing them to new files. Copying files only is supported when: * source_uris are in the same external storage system as the destination table but they do not overlap with storage_uri of the destination table. * source_format is the same file format as the destination table. * destination_table is an existing BigLake managed table. Its schema does not have default value expression. It schema does not have type parameters other than precision and scale. * No options other than the above are specified.
	Destinationtableproperties DestinationTableProperties `json:"destinationTableProperties,omitempty"` // Properties for the destination table.
	Schemainline string `json:"schemaInline,omitempty"` // [Deprecated] The inline schema. For CSV schemas, specify as "Field1:Type1[,Field2:Type2]*". For example, "foo:STRING, bar:INTEGER, baz:FLOAT".
	Fielddelimiter string `json:"fieldDelimiter,omitempty"` // Optional. The separator character for fields in a CSV file. The separator is interpreted as a single byte. For files encoded in ISO-8859-1, any single character can be used as a separator. For files encoded in UTF-8, characters represented in decimal range 1-127 (U+0001-U+007F) can be used without any modification. UTF-8 characters encoded with multiple bytes (i.e. U+0080 and above) will have only the first byte used for separating fields. The remaining bytes will be treated as a part of the field. BigQuery also supports the escape sequence "\t" (U+0009) to specify a tab separator. The default value is comma (",", U+002C).
	Destinationtable TableReference `json:"destinationTable,omitempty"`
	Schemainlineformat string `json:"schemaInlineFormat,omitempty"` // [Deprecated] The format of the schemaInline property.
	Schemaupdateoptions []string `json:"schemaUpdateOptions,omitempty"` // Allows the schema of the destination table to be updated as a side effect of the load job if a schema is autodetected or supplied in the job configuration. Schema update options are supported in two cases: when writeDisposition is WRITE_APPEND; when writeDisposition is WRITE_TRUNCATE and the destination table is a partition of a table, specified by partition decorators. For normal tables, WRITE_TRUNCATE will always overwrite the schema. One or more of the following values are specified: * ALLOW_FIELD_ADDITION: allow adding a nullable field to the schema. * ALLOW_FIELD_RELAXATION: allow relaxing a required field in the original schema to nullable.
	Hivepartitioningoptions HivePartitioningOptions `json:"hivePartitioningOptions,omitempty"` // Options for configuring hive partitioning detect.
	Skipleadingrows int `json:"skipLeadingRows,omitempty"` // Optional. The number of rows at the top of a CSV file that BigQuery will skip when loading the data. The default value is 0. This property is useful if you have header rows in the file that should be skipped. When autodetect is on, the behavior is the following: * skipLeadingRows unspecified - Autodetect tries to detect headers in the first row. If they are not detected, the row is read as data. Otherwise data is read starting from the second row. * skipLeadingRows is 0 - Instructs autodetect that there are no headers and data should be read starting from the first row. * skipLeadingRows = N > 0 - Autodetect skips N-1 rows and tries to detect headers in row N. If headers are not detected, row N is just skipped. Otherwise row N is used to extract column names for the detected schema.
	Allowquotednewlines bool `json:"allowQuotedNewlines,omitempty"` // Indicates if BigQuery should allow quoted data sections that contain newline characters in a CSV file. The default value is false.
	Quote string `json:"quote,omitempty"` // Optional. The value that is used to quote data sections in a CSV file. BigQuery converts the string to ISO-8859-1 encoding, and then uses the first byte of the encoded string to split the data in its raw, binary state. The default value is a double-quote ('"'). If your data does not contain quoted sections, set the property value to an empty string. If your data contains quoted newline characters, you must also set the allowQuotedNewlines property to true. To include the specific quote character within a quoted value, precede it with an additional matching quote character. For example, if you want to escape the default character ' " ', use ' "" '. @default "
	Autodetect bool `json:"autodetect,omitempty"` // Optional. Indicates if we should automatically infer the options and schema for CSV and JSON sources.
	Useavrologicaltypes bool `json:"useAvroLogicalTypes,omitempty"` // Optional. If sourceFormat is set to "AVRO", indicates whether to interpret logical types as the corresponding BigQuery data type (for example, TIMESTAMP), instead of using the raw type (for example, INTEGER).
	Projectionfields []string `json:"projectionFields,omitempty"` // If sourceFormat is set to "DATASTORE_BACKUP", indicates which entity properties to load into BigQuery from a Cloud Datastore backup. Property names are case sensitive and must be top-level properties. If no properties are specified, BigQuery loads all properties. If any named property isn't found in the Cloud Datastore backup, an invalid error is returned in the job result.
	Createdisposition string `json:"createDisposition,omitempty"` // Optional. Specifies whether the job is allowed to create new tables. The following values are supported: * CREATE_IF_NEEDED: If the table does not exist, BigQuery creates the table. * CREATE_NEVER: The table must already exist. If it does not, a 'notFound' error is returned in the job result. The default value is CREATE_IF_NEEDED. Creation, truncation and append actions occur as one atomic update upon job completion.
	Filesetspectype string `json:"fileSetSpecType,omitempty"` // Optional. Specifies how source URIs are interpreted for constructing the file set to load. By default, source URIs are expanded against the underlying storage. You can also specify manifest files to control how the file set is constructed. This option is only applicable to object storage systems.
	Nullmarker string `json:"nullMarker,omitempty"` // Optional. Specifies a string that represents a null value in a CSV file. For example, if you specify "\N", BigQuery interprets "\N" as a null value when loading a CSV file. The default value is the empty string. If you set this property to a custom value, BigQuery throws an error if an empty string is present for all data types except for STRING and BYTE. For STRING and BYTE columns, BigQuery interprets the empty string as an empty value.
}

// Cluster represents the Cluster schema from the OpenAPI specification
type Cluster struct {
	Featurevalues []FeatureValue `json:"featureValues,omitempty"` // Values of highly variant features for this cluster.
	Centroidid string `json:"centroidId,omitempty"` // Centroid id.
	Count string `json:"count,omitempty"` // Count of training data rows that were assigned to this cluster.
}

// CategoryCount represents the CategoryCount schema from the OpenAPI specification
type CategoryCount struct {
	Category string `json:"category,omitempty"` // The name of category.
	Count string `json:"count,omitempty"` // The count of training samples matching the category within the cluster.
}

// InputDataChange represents the InputDataChange schema from the OpenAPI specification
type InputDataChange struct {
	Recordsreaddiffpercentage float32 `json:"recordsReadDiffPercentage,omitempty"` // Output only. Records read difference percentage compared to a previous run.
}

// RangePartitioning represents the RangePartitioning schema from the OpenAPI specification
type RangePartitioning struct {
	Field string `json:"field,omitempty"` // Required. [Experimental] The table is partitioned by this field. The field must be a top-level NULLABLE/REQUIRED field. The only supported type is INTEGER/INT64.
	RangeField map[string]interface{} `json:"range,omitempty"` // [Experimental] Defines the ranges for range partitioning.
}

// CategoricalValue represents the CategoricalValue schema from the OpenAPI specification
type CategoricalValue struct {
	Categorycounts []CategoryCount `json:"categoryCounts,omitempty"` // Counts of all categories for the categorical feature. If there are more than ten categories, we return top ten (by count) and return one more CategoryCount with category "_OTHER_" and count as aggregate counts of remaining categories.
}

// ArimaModelInfo represents the ArimaModelInfo schema from the OpenAPI specification
type ArimaModelInfo struct {
	Hasholidayeffect bool `json:"hasHolidayEffect,omitempty"` // If true, holiday_effect is a part of time series decomposition result.
	Seasonalperiods []string `json:"seasonalPeriods,omitempty"` // Seasonal periods. Repeated because multiple periods are supported for one time series.
	Hasspikesanddips bool `json:"hasSpikesAndDips,omitempty"` // If true, spikes_and_dips is a part of time series decomposition result.
	Hasstepchanges bool `json:"hasStepChanges,omitempty"` // If true, step_changes is a part of time series decomposition result.
	Timeseriesids []string `json:"timeSeriesIds,omitempty"` // The tuple of time_series_ids identifying this time series. It will be one of the unique tuples of values present in the time_series_id_columns specified during ARIMA model training. Only present when time_series_id_columns training option was used and the order of values here are same as the order of time_series_id_columns.
	Arimafittingmetrics ArimaFittingMetrics `json:"arimaFittingMetrics,omitempty"` // ARIMA model fitting metrics.
	Hasdrift bool `json:"hasDrift,omitempty"` // Whether Arima model fitted with drift or not. It is always false when d is not 1.
	Nonseasonalorder ArimaOrder `json:"nonSeasonalOrder,omitempty"` // Arima order, can be used for both non-seasonal and seasonal parts.
	Timeseriesid string `json:"timeSeriesId,omitempty"` // The time_series_id value for this time series. It will be one of the unique values from the time_series_id_column specified during ARIMA model training. Only present when time_series_id_column training option was used.
	Arimacoefficients ArimaCoefficients `json:"arimaCoefficients,omitempty"` // Arima coefficients.
}

// AggregationThresholdPolicy represents the AggregationThresholdPolicy schema from the OpenAPI specification
type AggregationThresholdPolicy struct {
	Privacyunitcolumns []string `json:"privacyUnitColumns,omitempty"` // Optional. The privacy unit column(s) associated with this policy. For now, only one column per data source object (table, view) is allowed as a privacy unit column. Representing as a repeated field in metadata for extensibility to multiple columns in future. Duplicates and Repeated struct fields are not allowed. For nested fields, use dot notation ("outer.inner")
	Threshold string `json:"threshold,omitempty"` // Optional. The threshold for the "aggregation threshold" policy.
}

// QueryInfo represents the QueryInfo schema from the OpenAPI specification
type QueryInfo struct {
	Optimizationdetails map[string]interface{} `json:"optimizationDetails,omitempty"` // Output only. Information about query optimizations.
}

// ExplainQueryStage represents the ExplainQueryStage schema from the OpenAPI specification
type ExplainQueryStage struct {
	Readmsavg string `json:"readMsAvg,omitempty"` // Milliseconds the average shard spent reading input.
	Computemode string `json:"computeMode,omitempty"` // Output only. Compute mode for this stage.
	Waitratiomax float64 `json:"waitRatioMax,omitempty"` // Relative amount of time the slowest shard spent waiting to be scheduled.
	Writemsavg string `json:"writeMsAvg,omitempty"` // Milliseconds the average shard spent on writing output.
	Waitmsavg string `json:"waitMsAvg,omitempty"` // Milliseconds the average shard spent waiting to be scheduled.
	Readratioavg float64 `json:"readRatioAvg,omitempty"` // Relative amount of time the average shard spent reading input.
	Status string `json:"status,omitempty"` // Current status for this stage.
	Recordsread string `json:"recordsRead,omitempty"` // Number of records read into the stage.
	Name string `json:"name,omitempty"` // Human-readable name for the stage.
	Slotms string `json:"slotMs,omitempty"` // Slot-milliseconds used by the stage.
	Writeratiomax float64 `json:"writeRatioMax,omitempty"` // Relative amount of time the slowest shard spent on writing output.
	Startms string `json:"startMs,omitempty"` // Stage start time represented as milliseconds since the epoch.
	Id string `json:"id,omitempty"` // Unique ID for the stage within the plan.
	Readmsmax string `json:"readMsMax,omitempty"` // Milliseconds the slowest shard spent reading input.
	Inputstages []string `json:"inputStages,omitempty"` // IDs for stages that are inputs to this stage.
	Endms string `json:"endMs,omitempty"` // Stage end time represented as milliseconds since the epoch.
	Readratiomax float64 `json:"readRatioMax,omitempty"` // Relative amount of time the slowest shard spent reading input.
	Computemsmax string `json:"computeMsMax,omitempty"` // Milliseconds the slowest shard spent on CPU-bound tasks.
	Completedparallelinputs string `json:"completedParallelInputs,omitempty"` // Number of parallel input segments completed.
	Shuffleoutputbytesspilled string `json:"shuffleOutputBytesSpilled,omitempty"` // Total number of bytes written to shuffle and spilled to disk.
	Parallelinputs string `json:"parallelInputs,omitempty"` // Number of parallel input segments to be processed
	Waitratioavg float64 `json:"waitRatioAvg,omitempty"` // Relative amount of time the average shard spent waiting to be scheduled.
	Recordswritten string `json:"recordsWritten,omitempty"` // Number of records written by the stage.
	Shuffleoutputbytes string `json:"shuffleOutputBytes,omitempty"` // Total number of bytes written to shuffle.
	Writemsmax string `json:"writeMsMax,omitempty"` // Milliseconds the slowest shard spent on writing output.
	Computeratioavg float64 `json:"computeRatioAvg,omitempty"` // Relative amount of time the average shard spent on CPU-bound tasks.
	Waitmsmax string `json:"waitMsMax,omitempty"` // Milliseconds the slowest shard spent waiting to be scheduled.
	Steps []ExplainQueryStep `json:"steps,omitempty"` // List of operations within the stage in dependency order (approximately chronological).
	Computemsavg string `json:"computeMsAvg,omitempty"` // Milliseconds the average shard spent on CPU-bound tasks.
	Writeratioavg float64 `json:"writeRatioAvg,omitempty"` // Relative amount of time the average shard spent on writing output.
	Computeratiomax float64 `json:"computeRatioMax,omitempty"` // Relative amount of time the slowest shard spent on CPU-bound tasks.
}

// JobStatistics5 represents the JobStatistics5 schema from the OpenAPI specification
type JobStatistics5 struct {
	Copiedrows string `json:"copiedRows,omitempty"` // Output only. Number of rows copied to the destination table.
	Copiedlogicalbytes string `json:"copiedLogicalBytes,omitempty"` // Output only. Number of logical bytes copied to the destination table.
}

// JobStatus represents the JobStatus schema from the OpenAPI specification
type JobStatus struct {
	Errorresult ErrorProto `json:"errorResult,omitempty"` // Error details.
	Errors []ErrorProto `json:"errors,omitempty"` // Output only. The first errors encountered during the running of the job. The final message includes the number of errors that caused the process to stop. Errors here do not necessarily mean that the job has not completed or was unsuccessful.
	State string `json:"state,omitempty"` // Output only. Running state of the job. Valid states include 'PENDING', 'RUNNING', and 'DONE'.
}

// SearchStatistics represents the SearchStatistics schema from the OpenAPI specification
type SearchStatistics struct {
	Indexunusedreasons []IndexUnusedReason `json:"indexUnusedReasons,omitempty"` // When `indexUsageMode` is `UNUSED` or `PARTIALLY_USED`, this field explains why indexes were not used in all or part of the search query. If `indexUsageMode` is `FULLY_USED`, this field is not populated.
	Indexusagemode string `json:"indexUsageMode,omitempty"` // Specifies the index usage mode for the query.
}

// BqmlIterationResult represents the BqmlIterationResult schema from the OpenAPI specification
type BqmlIterationResult struct {
	Durationms string `json:"durationMs,omitempty"` // Deprecated.
	Evalloss float64 `json:"evalLoss,omitempty"` // Deprecated.
	Index int `json:"index,omitempty"` // Deprecated.
	Learnrate float64 `json:"learnRate,omitempty"` // Deprecated.
	Trainingloss float64 `json:"trainingLoss,omitempty"` // Deprecated.
}

// ScriptStatistics represents the ScriptStatistics schema from the OpenAPI specification
type ScriptStatistics struct {
	Evaluationkind string `json:"evaluationKind,omitempty"` // Whether this child job was a statement or expression.
	Stackframes []ScriptStackFrame `json:"stackFrames,omitempty"` // Stack trace showing the line/column/procedure name of each frame on the stack at the point where the current evaluation happened. The leaf frame is first, the primary script is last. Never empty.
}

// TableSchema represents the TableSchema schema from the OpenAPI specification
type TableSchema struct {
	Fields []TableFieldSchema `json:"fields,omitempty"` // Describes the fields in a table.
}

// JobConfiguration represents the JobConfiguration schema from the OpenAPI specification
type JobConfiguration struct {
	Load JobConfigurationLoad `json:"load,omitempty"` // JobConfigurationLoad contains the configuration properties for loading data into a destination table.
	Query JobConfigurationQuery `json:"query,omitempty"` // JobConfigurationQuery configures a BigQuery query job.
	CopyField JobConfigurationTableCopy `json:"copy,omitempty"` // JobConfigurationTableCopy configures a job that copies data from one table to another. For more information on copying tables, see [Copy a table](https://cloud.google.com/bigquery/docs/managing-tables#copy-table).
	Dryrun bool `json:"dryRun,omitempty"` // Optional. If set, don't actually run this job. A valid query will return a mostly empty response with some processing statistics, while an invalid query will return the same error it would if it wasn't a dry run. Behavior of non-query jobs is undefined.
	Extract JobConfigurationExtract `json:"extract,omitempty"` // JobConfigurationExtract configures a job that exports data from a BigQuery table into Google Cloud Storage.
	Jobtimeoutms string `json:"jobTimeoutMs,omitempty"` // Optional. Job timeout in milliseconds. If this time limit is exceeded, BigQuery might attempt to stop the job.
	Jobtype string `json:"jobType,omitempty"` // Output only. The type of the job. Can be QUERY, LOAD, EXTRACT, COPY or UNKNOWN.
	Labels map[string]interface{} `json:"labels,omitempty"` // The labels associated with this job. You can use these to organize and group your jobs. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label values are optional. Label keys must start with a letter and each label in the list must have a different key.
}

// TableFieldSchema represents the TableFieldSchema schema from the OpenAPI specification
type TableFieldSchema struct {
	Collation string `json:"collation,omitempty"` // Optional. Field collation can be set only when the type of field is STRING. The following values are supported: * 'und:ci': undetermined locale, case insensitive. * '': empty string. Default to case-sensitive behavior.
	Name string `json:"name,omitempty"` // Required. The field name. The name must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_), and must start with a letter or underscore. The maximum length is 300 characters.
	Precision string `json:"precision,omitempty"` // Optional. Precision (maximum number of total digits in base 10) and scale (maximum number of digits in the fractional part in base 10) constraints for values of this field for NUMERIC or BIGNUMERIC. It is invalid to set precision or scale if type ≠ "NUMERIC" and ≠ "BIGNUMERIC". If precision and scale are not specified, no value range constraint is imposed on this field insofar as values are permitted by the type. Values of this NUMERIC or BIGNUMERIC field must be in this range when: * Precision (P) and scale (S) are specified: [-10P-S + 10-S, 10P-S - 10-S] * Precision (P) is specified but not scale (and thus scale is interpreted to be equal to zero): [-10P + 1, 10P - 1]. Acceptable values for precision and scale if both are specified: * If type = "NUMERIC": 1 ≤ precision - scale ≤ 29 and 0 ≤ scale ≤ 9. * If type = "BIGNUMERIC": 1 ≤ precision - scale ≤ 38 and 0 ≤ scale ≤ 38. Acceptable values for precision if only precision is specified but not scale (and thus scale is interpreted to be equal to zero): * If type = "NUMERIC": 1 ≤ precision ≤ 29. * If type = "BIGNUMERIC": 1 ≤ precision ≤ 38. If scale is specified but not precision, then it is invalid.
	Rangeelementtype map[string]interface{} `json:"rangeElementType,omitempty"` // Represents the type of a field element.
	Categories map[string]interface{} `json:"categories,omitempty"` // Deprecated.
	Defaultvalueexpression string `json:"defaultValueExpression,omitempty"` // Optional. A SQL expression to specify the [default value] (https://cloud.google.com/bigquery/docs/default-values) for this field.
	Roundingmode string `json:"roundingMode,omitempty"` // Optional. Specifies the rounding mode to be used when storing values of NUMERIC and BIGNUMERIC type.
	Description string `json:"description,omitempty"` // Optional. The field description. The maximum length is 1,024 characters.
	Fields []TableFieldSchema `json:"fields,omitempty"` // Optional. Describes the nested schema fields if the type property is set to RECORD.
	Maxlength string `json:"maxLength,omitempty"` // Optional. Maximum length of values of this field for STRINGS or BYTES. If max_length is not specified, no maximum length constraint is imposed on this field. If type = "STRING", then max_length represents the maximum UTF-8 length of strings in this field. If type = "BYTES", then max_length represents the maximum number of bytes in this field. It is invalid to set this field if type ≠ "STRING" and ≠ "BYTES".
	Policytags map[string]interface{} `json:"policyTags,omitempty"` // Optional. The policy tags attached to this field, used for field-level access control. If not set, defaults to empty policy_tags.
	Mode string `json:"mode,omitempty"` // Optional. The field mode. Possible values include NULLABLE, REQUIRED and REPEATED. The default value is NULLABLE.
	Scale string `json:"scale,omitempty"` // Optional. See documentation for precision.
	TypeField string `json:"type,omitempty"` // Required. The field data type. Possible values include: * STRING * BYTES * INTEGER (or INT64) * FLOAT (or FLOAT64) * BOOLEAN (or BOOL) * TIMESTAMP * DATE * TIME * DATETIME * GEOGRAPHY * NUMERIC * BIGNUMERIC * JSON * RECORD (or STRUCT) Use of RECORD/STRUCT indicates that the field contains a nested schema.
}

// ExternalServiceCost represents the ExternalServiceCost schema from the OpenAPI specification
type ExternalServiceCost struct {
	Reservedslotcount string `json:"reservedSlotCount,omitempty"` // Non-preemptable reserved slots used for external job. For example, reserved slots for Cloua AI Platform job are the VM usages converted to BigQuery slot with equivalent mount of price.
	Slotms string `json:"slotMs,omitempty"` // External service cost in terms of bigquery slot milliseconds.
	Bytesbilled string `json:"bytesBilled,omitempty"` // External service cost in terms of bigquery bytes billed.
	Bytesprocessed string `json:"bytesProcessed,omitempty"` // External service cost in terms of bigquery bytes processed.
	Externalservice string `json:"externalService,omitempty"` // External service name.
}

// Policy represents the Policy schema from the OpenAPI specification
type Policy struct {
	Auditconfigs []AuditConfig `json:"auditConfigs,omitempty"` // Specifies cloud audit logging configuration for this policy.
	Bindings []Binding `json:"bindings,omitempty"` // Associates a list of `members`, or principals, with a `role`. Optionally, may specify a `condition` that determines how and when the `bindings` are applied. Each of the `bindings` must contain at least one principal. The `bindings` in a `Policy` can refer to up to 1,500 principals; up to 250 of these principals can be Google groups. Each occurrence of a principal counts towards these limits. For example, if the `bindings` grant 50 different roles to `user:alice@example.com`, and not to any other principal, then you can add another 1,450 principals to the `bindings` in the `Policy`.
	Etag string `json:"etag,omitempty"` // `etag` is used for optimistic concurrency control as a way to help prevent simultaneous updates of a policy from overwriting each other. It is strongly suggested that systems make use of the `etag` in the read-modify-write cycle to perform policy updates in order to avoid race conditions: An `etag` is returned in the response to `getIamPolicy`, and systems are expected to put that etag in the request to `setIamPolicy` to ensure that their change will be applied to the same version of the policy. **Important:** If you use IAM Conditions, you must include the `etag` field whenever you call `setIamPolicy`. If you omit this field, then IAM allows you to overwrite a version `3` policy with a version `1` policy, and all of the conditions in the version `3` policy are lost.
	Version int `json:"version,omitempty"` // Specifies the format of the policy. Valid values are `0`, `1`, and `3`. Requests that specify an invalid value are rejected. Any operation that affects conditional role bindings must specify version `3`. This requirement applies to the following operations: * Getting a policy that includes a conditional role binding * Adding a conditional role binding to a policy * Changing a conditional role binding in a policy * Removing any role binding, with or without a condition, from a policy that includes conditions **Important:** If you use IAM Conditions, you must include the `etag` field whenever you call `setIamPolicy`. If you omit this field, then IAM allows you to overwrite a version `3` policy with a version `1` policy, and all of the conditions in the version `3` policy are lost. If a policy does not include any conditions, operations on that policy may specify any valid version or leave the field unset. To learn which resources support conditions in their IAM policies, see the [IAM documentation](https://cloud.google.com/iam/help/conditions/resource-policies).
}

// GetPolicyOptions represents the GetPolicyOptions schema from the OpenAPI specification
type GetPolicyOptions struct {
	Requestedpolicyversion int `json:"requestedPolicyVersion,omitempty"` // Optional. The maximum policy version that will be used to format the policy. Valid values are 0, 1, and 3. Requests specifying an invalid value will be rejected. Requests for policies with any conditional role bindings must specify version 3. Policies with no conditional role bindings may specify any valid value or leave the field unset. The policy in the response might use the policy version that you specified, or it might use a lower policy version. For example, if you specify version 3, but the policy has no conditional role bindings, the response uses version 1. To learn which resources support conditions in their IAM policies, see the [IAM documentation](https://cloud.google.com/iam/help/conditions/resource-policies).
}

// IntHparamSearchSpace represents the IntHparamSearchSpace schema from the OpenAPI specification
type IntHparamSearchSpace struct {
	Candidates IntCandidates `json:"candidates,omitempty"` // Discrete candidates of an int hyperparameter.
	RangeField IntRange `json:"range,omitempty"` // Range of an int hyperparameter.
}

// CsvOptions represents the CsvOptions schema from the OpenAPI specification
type CsvOptions struct {
	Encoding string `json:"encoding,omitempty"` // Optional. The character encoding of the data. The supported values are UTF-8, ISO-8859-1, UTF-16BE, UTF-16LE, UTF-32BE, and UTF-32LE. The default value is UTF-8. BigQuery decodes the data after the raw, binary data has been split using the values of the quote and fieldDelimiter properties.
	Fielddelimiter string `json:"fieldDelimiter,omitempty"` // Optional. The separator character for fields in a CSV file. The separator is interpreted as a single byte. For files encoded in ISO-8859-1, any single character can be used as a separator. For files encoded in UTF-8, characters represented in decimal range 1-127 (U+0001-U+007F) can be used without any modification. UTF-8 characters encoded with multiple bytes (i.e. U+0080 and above) will have only the first byte used for separating fields. The remaining bytes will be treated as a part of the field. BigQuery also supports the escape sequence "\t" (U+0009) to specify a tab separator. The default value is comma (",", U+002C).
	Nullmarker string `json:"nullMarker,omitempty"` // [Optional] A custom string that will represent a NULL value in CSV import data.
	Preserveasciicontrolcharacters bool `json:"preserveAsciiControlCharacters,omitempty"` // Optional. Indicates if the embedded ASCII control characters (the first 32 characters in the ASCII-table, from '\x00' to '\x1F') are preserved.
	Quote string `json:"quote,omitempty"` // Optional. The value that is used to quote data sections in a CSV file. BigQuery converts the string to ISO-8859-1 encoding, and then uses the first byte of the encoded string to split the data in its raw, binary state. The default value is a double-quote ("). If your data does not contain quoted sections, set the property value to an empty string. If your data contains quoted newline characters, you must also set the allowQuotedNewlines property to true. To include the specific quote character within a quoted value, precede it with an additional matching quote character. For example, if you want to escape the default character ' " ', use ' "" '.
	Skipleadingrows string `json:"skipLeadingRows,omitempty"` // Optional. The number of rows at the top of a CSV file that BigQuery will skip when reading the data. The default value is 0. This property is useful if you have header rows in the file that should be skipped. When autodetect is on, the behavior is the following: * skipLeadingRows unspecified - Autodetect tries to detect headers in the first row. If they are not detected, the row is read as data. Otherwise data is read starting from the second row. * skipLeadingRows is 0 - Instructs autodetect that there are no headers and data should be read starting from the first row. * skipLeadingRows = N > 0 - Autodetect skips N-1 rows and tries to detect headers in row N. If headers are not detected, row N is just skipped. Otherwise row N is used to extract column names for the detected schema.
	Allowjaggedrows bool `json:"allowJaggedRows,omitempty"` // Optional. Indicates if BigQuery should accept rows that are missing trailing optional columns. If true, BigQuery treats missing trailing columns as null values. If false, records with missing trailing columns are treated as bad records, and if there are too many bad records, an invalid error is returned in the job result. The default value is false.
	Allowquotednewlines bool `json:"allowQuotedNewlines,omitempty"` // Optional. Indicates if BigQuery should allow quoted data sections that contain newline characters in a CSV file. The default value is false.
}

// DmlStatistics represents the DmlStatistics schema from the OpenAPI specification
type DmlStatistics struct {
	Insertedrowcount string `json:"insertedRowCount,omitempty"` // Output only. Number of inserted Rows. Populated by DML INSERT and MERGE statements
	Updatedrowcount string `json:"updatedRowCount,omitempty"` // Output only. Number of updated Rows. Populated by DML UPDATE and MERGE statements.
	Deletedrowcount string `json:"deletedRowCount,omitempty"` // Output only. Number of deleted Rows. populated by DML DELETE, MERGE and TRUNCATE statements.
}

// HparamTuningTrial represents the HparamTuningTrial schema from the OpenAPI specification
type HparamTuningTrial struct {
	Evaluationmetrics EvaluationMetrics `json:"evaluationMetrics,omitempty"` // Evaluation metrics of a model. These are either computed on all training data or just the eval data based on whether eval data was used during training. These are not present for imported models.
	Hparams TrainingOptions `json:"hparams,omitempty"` // Options used in model training.
	Endtimems string `json:"endTimeMs,omitempty"` // Ending time of the trial.
	Status string `json:"status,omitempty"` // The status of the trial.
	Hparamtuningevaluationmetrics EvaluationMetrics `json:"hparamTuningEvaluationMetrics,omitempty"` // Evaluation metrics of a model. These are either computed on all training data or just the eval data based on whether eval data was used during training. These are not present for imported models.
	Trainingloss float64 `json:"trainingLoss,omitempty"` // Loss computed on the training data at the end of trial.
	Trialid string `json:"trialId,omitempty"` // 1-based index of the trial.
	Errormessage string `json:"errorMessage,omitempty"` // Error message for FAILED and INFEASIBLE trial.
	Starttimems string `json:"startTimeMs,omitempty"` // Starting time of the trial.
	Evalloss float64 `json:"evalLoss,omitempty"` // Loss computed on the eval data at the end of trial.
}

// AvroOptions represents the AvroOptions schema from the OpenAPI specification
type AvroOptions struct {
	Useavrologicaltypes bool `json:"useAvroLogicalTypes,omitempty"` // Optional. If sourceFormat is set to "AVRO", indicates whether to interpret logical types as the corresponding BigQuery data type (for example, TIMESTAMP), instead of using the raw type (for example, INTEGER).
}

// UserDefinedFunctionResource represents the UserDefinedFunctionResource schema from the OpenAPI specification
type UserDefinedFunctionResource struct {
	Inlinecode string `json:"inlineCode,omitempty"` // [Pick one] An inline resource that contains code for a user-defined function (UDF). Providing a inline code resource is equivalent to providing a URI for a file containing the same code.
	Resourceuri string `json:"resourceUri,omitempty"` // [Pick one] A code resource to load from a Google Cloud Storage URI (gs://bucket/path).
}

// TableCell represents the TableCell schema from the OpenAPI specification
type TableCell struct {
	V interface{} `json:"v,omitempty"`
}

// ParquetOptions represents the ParquetOptions schema from the OpenAPI specification
type ParquetOptions struct {
	Enumasstring bool `json:"enumAsString,omitempty"` // Optional. Indicates whether to infer Parquet ENUM logical type as STRING instead of BYTES by default.
	Enablelistinference bool `json:"enableListInference,omitempty"` // Optional. Indicates whether to use schema inference specifically for Parquet LIST logical type.
}

// ModelDefinition represents the ModelDefinition schema from the OpenAPI specification
type ModelDefinition struct {
	Trainingruns []BqmlTrainingRun `json:"trainingRuns,omitempty"` // Deprecated.
	Modeloptions map[string]interface{} `json:"modelOptions,omitempty"` // Deprecated.
}

// IntRange represents the IntRange schema from the OpenAPI specification
type IntRange struct {
	Max string `json:"max,omitempty"` // Max value of the int parameter.
	Min string `json:"min,omitempty"` // Min value of the int parameter.
}

// DoubleCandidates represents the DoubleCandidates schema from the OpenAPI specification
type DoubleCandidates struct {
	Candidates []float64 `json:"candidates,omitempty"` // Candidates for the double parameter in increasing order.
}

// QueryTimelineSample represents the QueryTimelineSample schema from the OpenAPI specification
type QueryTimelineSample struct {
	Pendingunits string `json:"pendingUnits,omitempty"` // Total units of work remaining for the query. This number can be revised (increased or decreased) while the query is running.
	Totalslotms string `json:"totalSlotMs,omitempty"` // Cumulative slot-ms consumed by the query.
	Activeunits string `json:"activeUnits,omitempty"` // Total number of active workers. This does not correspond directly to slot usage. This is the largest value observed since the last sample.
	Completedunits string `json:"completedUnits,omitempty"` // Total parallel units of work completed by this query.
	Elapsedms string `json:"elapsedMs,omitempty"` // Milliseconds elapsed since the start of query execution.
	Estimatedrunnableunits string `json:"estimatedRunnableUnits,omitempty"` // Units of work that can be scheduled immediately. Providing additional slots for these units of work will accelerate the query, if no other query in the reservation needs additional slots.
}

// PrincipalComponentInfo represents the PrincipalComponentInfo schema from the OpenAPI specification
type PrincipalComponentInfo struct {
	Explainedvariance float64 `json:"explainedVariance,omitempty"` // Explained variance by this principal component, which is simply the eigenvalue.
	Explainedvarianceratio float64 `json:"explainedVarianceRatio,omitempty"` // Explained_variance over the total explained variance.
	Principalcomponentid string `json:"principalComponentId,omitempty"` // Id of the principal component.
	Cumulativeexplainedvarianceratio float64 `json:"cumulativeExplainedVarianceRatio,omitempty"` // The explained_variance is pre-ordered in the descending order to compute the cumulative explained variance ratio.
}

// RoutineReference represents the RoutineReference schema from the OpenAPI specification
type RoutineReference struct {
	Routineid string `json:"routineId,omitempty"` // Required. The ID of the routine. The ID must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_). The maximum length is 256 characters.
	Datasetid string `json:"datasetId,omitempty"` // Required. The ID of the dataset containing this routine.
	Projectid string `json:"projectId,omitempty"` // Required. The ID of the project containing this routine.
}

// DoubleHparamSearchSpace represents the DoubleHparamSearchSpace schema from the OpenAPI specification
type DoubleHparamSearchSpace struct {
	Candidates DoubleCandidates `json:"candidates,omitempty"` // Discrete candidates of a double hyperparameter.
	RangeField DoubleRange `json:"range,omitempty"` // Range of a double hyperparameter.
}

// DatasetReference represents the DatasetReference schema from the OpenAPI specification
type DatasetReference struct {
	Projectid string `json:"projectId,omitempty"` // Optional. The ID of the project containing this dataset.
	Datasetid string `json:"datasetId,omitempty"` // Required. A unique ID for this dataset, without the project name. The ID must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_). The maximum length is 1,024 characters.
}

// TransformColumn represents the TransformColumn schema from the OpenAPI specification
type TransformColumn struct {
	TypeField StandardSqlDataType `json:"type,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
	Name string `json:"name,omitempty"` // Output only. Name of the column.
	Transformsql string `json:"transformSql,omitempty"` // Output only. The SQL expression used in the column transform.
}

// DatasetList represents the DatasetList schema from the OpenAPI specification
type DatasetList struct {
	Unreachable []string `json:"unreachable,omitempty"` // A list of skipped locations that were unreachable. For more information about BigQuery locations, see: https://cloud.google.com/bigquery/docs/locations. Example: "europe-west5"
	Datasets []map[string]interface{} `json:"datasets,omitempty"` // An array of the dataset resources in the project. Each resource contains basic information. For full information about a particular dataset resource, use the Datasets: get method. This property is omitted when there are no datasets in the project.
	Etag string `json:"etag,omitempty"` // Output only. A hash value of the results page. You can use this property to determine if the page has changed since the last request.
	Kind string `json:"kind,omitempty"` // Output only. The resource type. This property always returns the value "bigquery#datasetList"
	Nextpagetoken string `json:"nextPageToken,omitempty"` // A token that can be used to request the next results page. This property is omitted on the final results page.
}

// MaterializedView represents the MaterializedView schema from the OpenAPI specification
type MaterializedView struct {
	Chosen bool `json:"chosen,omitempty"` // Whether the materialized view is chosen for the query. A materialized view can be chosen to rewrite multiple parts of the same query. If a materialized view is chosen to rewrite any part of the query, then this field is true, even if the materialized view was not chosen to rewrite others parts.
	Estimatedbytessaved string `json:"estimatedBytesSaved,omitempty"` // If present, specifies a best-effort estimation of the bytes saved by using the materialized view rather than its base tables.
	Rejectedreason string `json:"rejectedReason,omitempty"` // If present, specifies the reason why the materialized view was not chosen for the query.
	Tablereference TableReference `json:"tableReference,omitempty"`
}

// Argument represents the Argument schema from the OpenAPI specification
type Argument struct {
	Mode string `json:"mode,omitempty"` // Optional. Specifies whether the argument is input or output. Can be set for procedures only.
	Name string `json:"name,omitempty"` // Optional. The name of this argument. Can be absent for function return argument.
	Argumentkind string `json:"argumentKind,omitempty"` // Optional. Defaults to FIXED_TYPE.
	Datatype StandardSqlDataType `json:"dataType,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
	Isaggregate bool `json:"isAggregate,omitempty"` // Optional. Whether the argument is an aggregate function parameter. Must be Unset for routine types other than AGGREGATE_FUNCTION. For AGGREGATE_FUNCTION, if set to false, it is equivalent to adding "NOT AGGREGATE" clause in DDL; Otherwise, it is equivalent to omitting "NOT AGGREGATE" clause in DDL.
}

// IntArrayHparamSearchSpace represents the IntArrayHparamSearchSpace schema from the OpenAPI specification
type IntArrayHparamSearchSpace struct {
	Candidates []IntArray `json:"candidates,omitempty"` // Candidates for the int array parameter.
}

// TimePartitioning represents the TimePartitioning schema from the OpenAPI specification
type TimePartitioning struct {
	Expirationms string `json:"expirationMs,omitempty"` // Optional. Number of milliseconds for which to keep the storage for a partition. A wrapper is used here because 0 is an invalid value.
	Field string `json:"field,omitempty"` // Optional. If not set, the table is partitioned by pseudo column '_PARTITIONTIME'; if set, the table is partitioned by this field. The field must be a top-level TIMESTAMP or DATE field. Its mode must be NULLABLE or REQUIRED. A wrapper is used here because an empty string is an invalid value.
	Requirepartitionfilter bool `json:"requirePartitionFilter,omitempty"` // If set to true, queries over this table require a partition filter that can be used for partition elimination to be specified. This field is deprecated; please set the field with the same name on the table itself instead. This field needs a wrapper because we want to output the default value, false, if the user explicitly set it.
	TypeField string `json:"type,omitempty"` // Required. The supported types are DAY, HOUR, MONTH, and YEAR, which will generate one partition per day, hour, month, and year, respectively.
}

// Row represents the Row schema from the OpenAPI specification
type Row struct {
	Actuallabel string `json:"actualLabel,omitempty"` // The original label of this row.
	Entries []Entry `json:"entries,omitempty"` // Info describing predicted label distribution.
}

// GoogleSheetsOptions represents the GoogleSheetsOptions schema from the OpenAPI specification
type GoogleSheetsOptions struct {
	RangeField string `json:"range,omitempty"` // Optional. Range of a sheet to query from. Only used when non-empty. Typical format: sheet_name!top_left_cell_id:bottom_right_cell_id For example: sheet1!A1:B20
	Skipleadingrows string `json:"skipLeadingRows,omitempty"` // Optional. The number of rows at the top of a sheet that BigQuery will skip when reading the data. The default value is 0. This property is useful if you have header rows that should be skipped. When autodetect is on, the behavior is the following: * skipLeadingRows unspecified - Autodetect tries to detect headers in the first row. If they are not detected, the row is read as data. Otherwise data is read starting from the second row. * skipLeadingRows is 0 - Instructs autodetect that there are no headers and data should be read starting from the first row. * skipLeadingRows = N > 0 - Autodetect skips N-1 rows and tries to detect headers in row N. If headers are not detected, row N is just skipped. Otherwise row N is used to extract column names for the detected schema.
}

// Binding represents the Binding schema from the OpenAPI specification
type Binding struct {
	Members []string `json:"members,omitempty"` // Specifies the principals requesting access for a Google Cloud resource. `members` can have the following values: * `allUsers`: A special identifier that represents anyone who is on the internet; with or without a Google account. * `allAuthenticatedUsers`: A special identifier that represents anyone who is authenticated with a Google account or a service account. Does not include identities that come from external identity providers (IdPs) through identity federation. * `user:{emailid}`: An email address that represents a specific Google account. For example, `alice@example.com` . * `serviceAccount:{emailid}`: An email address that represents a Google service account. For example, `my-other-app@appspot.gserviceaccount.com`. * `serviceAccount:{projectid}.svc.id.goog[{namespace}/{kubernetes-sa}]`: An identifier for a [Kubernetes service account](https://cloud.google.com/kubernetes-engine/docs/how-to/kubernetes-service-accounts). For example, `my-project.svc.id.goog[my-namespace/my-kubernetes-sa]`. * `group:{emailid}`: An email address that represents a Google group. For example, `admins@example.com`. * `domain:{domain}`: The G Suite domain (primary) that represents all the users of that domain. For example, `google.com` or `example.com`. * `principal://iam.googleapis.com/locations/global/workforcePools/{pool_id}/subject/{subject_attribute_value}`: A single identity in a workforce identity pool. * `principalSet://iam.googleapis.com/locations/global/workforcePools/{pool_id}/group/{group_id}`: All workforce identities in a group. * `principalSet://iam.googleapis.com/locations/global/workforcePools/{pool_id}/attribute.{attribute_name}/{attribute_value}`: All workforce identities with a specific attribute value. * `principalSet://iam.googleapis.com/locations/global/workforcePools/{pool_id}/*`: All identities in a workforce identity pool. * `principal://iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/subject/{subject_attribute_value}`: A single identity in a workload identity pool. * `principalSet://iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/group/{group_id}`: A workload identity pool group. * `principalSet://iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/attribute.{attribute_name}/{attribute_value}`: All identities in a workload identity pool with a certain attribute. * `principalSet://iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/*`: All identities in a workload identity pool. * `deleted:user:{emailid}?uid={uniqueid}`: An email address (plus unique identifier) representing a user that has been recently deleted. For example, `alice@example.com?uid=123456789012345678901`. If the user is recovered, this value reverts to `user:{emailid}` and the recovered user retains the role in the binding. * `deleted:serviceAccount:{emailid}?uid={uniqueid}`: An email address (plus unique identifier) representing a service account that has been recently deleted. For example, `my-other-app@appspot.gserviceaccount.com?uid=123456789012345678901`. If the service account is undeleted, this value reverts to `serviceAccount:{emailid}` and the undeleted service account retains the role in the binding. * `deleted:group:{emailid}?uid={uniqueid}`: An email address (plus unique identifier) representing a Google group that has been recently deleted. For example, `admins@example.com?uid=123456789012345678901`. If the group is recovered, this value reverts to `group:{emailid}` and the recovered group retains the role in the binding. * `deleted:principal://iam.googleapis.com/locations/global/workforcePools/{pool_id}/subject/{subject_attribute_value}`: Deleted single identity in a workforce identity pool. For example, `deleted:principal://iam.googleapis.com/locations/global/workforcePools/my-pool-id/subject/my-subject-attribute-value`.
	Role string `json:"role,omitempty"` // Role that is assigned to the list of `members`, or principals. For example, `roles/viewer`, `roles/editor`, or `roles/owner`. For an overview of the IAM roles and permissions, see the [IAM documentation](https://cloud.google.com/iam/docs/roles-overview). For a list of the available pre-defined roles, see [here](https://cloud.google.com/iam/docs/understanding-roles).
	Condition Expr `json:"condition,omitempty"` // Represents a textual expression in the Common Expression Language (CEL) syntax. CEL is a C-like expression language. The syntax and semantics of CEL are documented at https://github.com/google/cel-spec. Example (Comparison): title: "Summary size limit" description: "Determines if a summary is less than 100 chars" expression: "document.summary.size() < 100" Example (Equality): title: "Requestor is owner" description: "Determines if requestor is the document owner" expression: "document.owner == request.auth.claims.email" Example (Logic): title: "Public documents" description: "Determine whether the document should be publicly visible" expression: "document.type != 'private' && document.type != 'internal'" Example (Data Manipulation): title: "Notification string" description: "Create a notification string with a timestamp." expression: "'New message received at ' + string(document.create_time)" The exact variables and functions that may be referenced within an expression are determined by the service that evaluates it. See the service documentation for additional information.
}

// JobStatistics represents the JobStatistics schema from the OpenAPI specification
type JobStatistics struct {
	Rowlevelsecuritystatistics RowLevelSecurityStatistics `json:"rowLevelSecurityStatistics,omitempty"` // Statistics for row-level security.
	Completionratio float64 `json:"completionRatio,omitempty"` // Output only. [TrustedTester] Job progress (0.0 -> 1.0) for LOAD and EXTRACT jobs.
	Sessioninfo SessionInfo `json:"sessionInfo,omitempty"` // [Preview] Information related to sessions.
	Quotadeferments []string `json:"quotaDeferments,omitempty"` // Output only. Quotas which delayed this job's start time.
	Finalexecutiondurationms string `json:"finalExecutionDurationMs,omitempty"` // Output only. The duration in milliseconds of the execution of the final attempt of this job, as BigQuery may internally re-attempt to execute the job.
	Endtime string `json:"endTime,omitempty"` // Output only. End time of this job, in milliseconds since the epoch. This field will be present whenever a job is in the DONE state.
	Transactioninfo TransactionInfo `json:"transactionInfo,omitempty"` // [Alpha] Information of a multi-statement transaction.
	Reservationusage []map[string]interface{} `json:"reservationUsage,omitempty"` // Output only. Job resource usage breakdown by reservation. This field reported misleading information and will no longer be populated.
	Totalbytesprocessed string `json:"totalBytesProcessed,omitempty"` // Output only. Total bytes processed for the job.
	Scriptstatistics ScriptStatistics `json:"scriptStatistics,omitempty"` // Job statistics specific to the child job of a script.
	Reservation_id string `json:"reservation_id,omitempty"` // Output only. Name of the primary reservation assigned to this job. Note that this could be different than reservations reported in the reservation usage field if parent reservations were used to execute this job.
	Load JobStatistics3 `json:"load,omitempty"` // Statistics for a load job.
	CopyField JobStatistics5 `json:"copy,omitempty"` // Statistics for a copy job.
	Totalslotms string `json:"totalSlotMs,omitempty"` // Output only. Slot-milliseconds for the job.
	Creationtime string `json:"creationTime,omitempty"` // Output only. Creation time of this job, in milliseconds since the epoch. This field will be present on all jobs.
	Datamaskingstatistics DataMaskingStatistics `json:"dataMaskingStatistics,omitempty"` // Statistics for data-masking.
	Extract JobStatistics4 `json:"extract,omitempty"` // Statistics for an extract job.
	Starttime string `json:"startTime,omitempty"` // Output only. Start time of this job, in milliseconds since the epoch. This field will be present when the job transitions from the PENDING state to either RUNNING or DONE.
	Numchildjobs string `json:"numChildJobs,omitempty"` // Output only. Number of child jobs executed.
	Parentjobid string `json:"parentJobId,omitempty"` // Output only. If this is a child job, specifies the job ID of the parent.
	Query JobStatistics2 `json:"query,omitempty"` // Statistics for a query job.
}

// GetQueryResultsResponse represents the GetQueryResultsResponse schema from the OpenAPI specification
type GetQueryResultsResponse struct {
	Totalrows string `json:"totalRows,omitempty"` // The total number of rows in the complete query result set, which can be more than the number of rows in this single page of results. Present only when the query completes successfully.
	Pagetoken string `json:"pageToken,omitempty"` // A token used for paging results. When this token is non-empty, it indicates additional results are available.
	Etag string `json:"etag,omitempty"` // A hash of this response.
	Jobcomplete bool `json:"jobComplete,omitempty"` // Whether the query has completed or not. If rows or totalRows are present, this will always be true. If this is false, totalRows will not be available.
	Rows []TableRow `json:"rows,omitempty"` // An object with as many results as can be contained within the maximum permitted reply size. To get any additional rows, you can call GetQueryResults and specify the jobReference returned above. Present only when the query completes successfully. The REST-based representation of this data leverages a series of JSON f,v objects for indicating fields and values.
	Cachehit bool `json:"cacheHit,omitempty"` // Whether the query result was fetched from the query cache.
	Jobreference JobReference `json:"jobReference,omitempty"` // A job reference is a fully qualified identifier for referring to a job.
	Numdmlaffectedrows string `json:"numDmlAffectedRows,omitempty"` // Output only. The number of rows affected by a DML statement. Present only for DML statements INSERT, UPDATE or DELETE.
	Totalbytesprocessed string `json:"totalBytesProcessed,omitempty"` // The total number of bytes processed for this query.
	Errors []ErrorProto `json:"errors,omitempty"` // Output only. The first errors or warnings encountered during the running of the job. The final message includes the number of errors that caused the process to stop. Errors here do not necessarily mean that the job has completed or was unsuccessful. For more information about error messages, see [Error messages](https://cloud.google.com/bigquery/docs/error-messages).
	Kind string `json:"kind,omitempty"` // The resource type of the response.
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
}

// ArimaResult represents the ArimaResult schema from the OpenAPI specification
type ArimaResult struct {
	Arimamodelinfo []ArimaModelInfo `json:"arimaModelInfo,omitempty"` // This message is repeated because there are multiple arima models fitted in auto-arima. For non-auto-arima model, its size is one.
	Seasonalperiods []string `json:"seasonalPeriods,omitempty"` // Seasonal periods. Repeated because multiple periods are supported for one time series.
}

// StagePerformanceStandaloneInsight represents the StagePerformanceStandaloneInsight schema from the OpenAPI specification
type StagePerformanceStandaloneInsight struct {
	Highcardinalityjoins []HighCardinalityJoin `json:"highCardinalityJoins,omitempty"` // Output only. High cardinality joins in the stage.
	Insufficientshufflequota bool `json:"insufficientShuffleQuota,omitempty"` // Output only. True if the stage has insufficient shuffle quota.
	Slotcontention bool `json:"slotContention,omitempty"` // Output only. True if the stage has a slot contention issue.
	Stageid string `json:"stageId,omitempty"` // Output only. The stage id that the insight mapped to.
	Bienginereasons []BiEngineReason `json:"biEngineReasons,omitempty"` // Output only. If present, the stage had the following reasons for being disqualified from BI Engine execution.
}

// StandardSqlStructType represents the StandardSqlStructType schema from the OpenAPI specification
type StandardSqlStructType struct {
	Fields []StandardSqlField `json:"fields,omitempty"` // Fields within the struct.
}

// GetIamPolicyRequest represents the GetIamPolicyRequest schema from the OpenAPI specification
type GetIamPolicyRequest struct {
	Options GetPolicyOptions `json:"options,omitempty"` // Encapsulates settings provided to GetIamPolicy.
}

// IntArray represents the IntArray schema from the OpenAPI specification
type IntArray struct {
	Elements []string `json:"elements,omitempty"` // Elements in the int array.
}

// BigQueryModelTraining represents the BigQueryModelTraining schema from the OpenAPI specification
type BigQueryModelTraining struct {
	Currentiteration int `json:"currentIteration,omitempty"` // Deprecated.
	Expectedtotaliterations string `json:"expectedTotalIterations,omitempty"` // Deprecated.
}

// RowLevelSecurityStatistics represents the RowLevelSecurityStatistics schema from the OpenAPI specification
type RowLevelSecurityStatistics struct {
	Rowlevelsecurityapplied bool `json:"rowLevelSecurityApplied,omitempty"` // Whether any accessed data was protected by row access policies.
}

// ArimaFittingMetrics represents the ArimaFittingMetrics schema from the OpenAPI specification
type ArimaFittingMetrics struct {
	Aic float64 `json:"aic,omitempty"` // AIC.
	Loglikelihood float64 `json:"logLikelihood,omitempty"` // Log-likelihood.
	Variance float64 `json:"variance,omitempty"` // Variance.
}

// AggregateClassificationMetrics represents the AggregateClassificationMetrics schema from the OpenAPI specification
type AggregateClassificationMetrics struct {
	Precision float64 `json:"precision,omitempty"` // Precision is the fraction of actual positive predictions that had positive actual labels. For multiclass this is a macro-averaged metric treating each class as a binary classifier.
	Recall float64 `json:"recall,omitempty"` // Recall is the fraction of actual positive labels that were given a positive prediction. For multiclass this is a macro-averaged metric.
	Rocauc float64 `json:"rocAuc,omitempty"` // Area Under a ROC Curve. For multiclass this is a macro-averaged metric.
	Threshold float64 `json:"threshold,omitempty"` // Threshold at which the metrics are computed. For binary classification models this is the positive class threshold. For multi-class classfication models this is the confidence threshold.
	Accuracy float64 `json:"accuracy,omitempty"` // Accuracy is the fraction of predictions given the correct label. For multiclass this is a micro-averaged metric.
	F1score float64 `json:"f1Score,omitempty"` // The F1 score is an average of recall and precision. For multiclass this is a macro-averaged metric.
	Logloss float64 `json:"logLoss,omitempty"` // Logarithmic Loss. For multiclass this is a macro-averaged metric.
}

// MlStatistics represents the MlStatistics schema from the OpenAPI specification
type MlStatistics struct {
	Modeltype string `json:"modelType,omitempty"` // Output only. The type of the model that is being trained.
	Trainingtype string `json:"trainingType,omitempty"` // Output only. Training type of the job.
	Hparamtrials []HparamTuningTrial `json:"hparamTrials,omitempty"` // Output only. Trials of a [hyperparameter tuning job](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) sorted by trial_id.
	Iterationresults []IterationResult `json:"iterationResults,omitempty"` // Results for all completed iterations. Empty for [hyperparameter tuning jobs](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview).
	Maxiterations string `json:"maxIterations,omitempty"` // Output only. Maximum number of iterations specified as max_iterations in the 'CREATE MODEL' query. The actual number of iterations may be less than this number due to early stop.
}

// StandardSqlTableType represents the StandardSqlTableType schema from the OpenAPI specification
type StandardSqlTableType struct {
	Columns []StandardSqlField `json:"columns,omitempty"` // The columns in this table type
}

// ConnectionProperty represents the ConnectionProperty schema from the OpenAPI specification
type ConnectionProperty struct {
	Value string `json:"value,omitempty"` // The value of the property to set.
	Key string `json:"key,omitempty"` // The key of the property to set.
}

// SetIamPolicyRequest represents the SetIamPolicyRequest schema from the OpenAPI specification
type SetIamPolicyRequest struct {
	Policy Policy `json:"policy,omitempty"` // An Identity and Access Management (IAM) policy, which specifies access controls for Google Cloud resources. A `Policy` is a collection of `bindings`. A `binding` binds one or more `members`, or principals, to a single `role`. Principals can be user accounts, service accounts, Google groups, and domains (such as G Suite). A `role` is a named list of permissions; each `role` can be an IAM predefined role or a user-created custom role. For some types of Google Cloud resources, a `binding` can also specify a `condition`, which is a logical expression that allows access to a resource only if the expression evaluates to `true`. A condition can add constraints based on attributes of the request, the resource, or both. To learn which resources support conditions in their IAM policies, see the [IAM documentation](https://cloud.google.com/iam/help/conditions/resource-policies). **JSON example:** ``` { "bindings": [ { "role": "roles/resourcemanager.organizationAdmin", "members": [ "user:mike@example.com", "group:admins@example.com", "domain:google.com", "serviceAccount:my-project-id@appspot.gserviceaccount.com" ] }, { "role": "roles/resourcemanager.organizationViewer", "members": [ "user:eve@example.com" ], "condition": { "title": "expirable access", "description": "Does not grant access after Sep 2020", "expression": "request.time < timestamp('2020-10-01T00:00:00.000Z')", } } ], "etag": "BwWWja0YfJA=", "version": 3 } ``` **YAML example:** ``` bindings: - members: - user:mike@example.com - group:admins@example.com - domain:google.com - serviceAccount:my-project-id@appspot.gserviceaccount.com role: roles/resourcemanager.organizationAdmin - members: - user:eve@example.com role: roles/resourcemanager.organizationViewer condition: title: expirable access description: Does not grant access after Sep 2020 expression: request.time < timestamp('2020-10-01T00:00:00.000Z') etag: BwWWja0YfJA= version: 3 ``` For a description of IAM and its features, see the [IAM documentation](https://cloud.google.com/iam/docs/).
	Updatemask string `json:"updateMask,omitempty"` // OPTIONAL: A FieldMask specifying which fields of the policy to modify. Only the fields in the mask will be modified. If no mask is provided, the following default mask is used: `paths: "bindings, etag"`
}

// MultiClassClassificationMetrics represents the MultiClassClassificationMetrics schema from the OpenAPI specification
type MultiClassClassificationMetrics struct {
	Aggregateclassificationmetrics AggregateClassificationMetrics `json:"aggregateClassificationMetrics,omitempty"` // Aggregate metrics for classification/classifier models. For multi-class models, the metrics are either macro-averaged or micro-averaged. When macro-averaged, the metrics are calculated for each label and then an unweighted average is taken of those values. When micro-averaged, the metric is calculated globally by counting the total number of correctly predicted rows.
	Confusionmatrixlist []ConfusionMatrix `json:"confusionMatrixList,omitempty"` // Confusion matrix at different thresholds.
}

// Routine represents the Routine schema from the OpenAPI specification
type Routine struct {
	Etag string `json:"etag,omitempty"` // Output only. A hash of this resource.
	Language string `json:"language,omitempty"` // Optional. Defaults to "SQL" if remote_function_options field is absent, not set otherwise.
	Definitionbody string `json:"definitionBody,omitempty"` // Required. The body of the routine. For functions, this is the expression in the AS clause. If language=SQL, it is the substring inside (but excluding) the parentheses. For example, for the function created with the following statement: `CREATE FUNCTION JoinLines(x string, y string) as (concat(x, "\n", y))` The definition_body is `concat(x, "\n", y)` (\n is not replaced with linebreak). If language=JAVASCRIPT, it is the evaluated string in the AS clause. For example, for the function created with the following statement: `CREATE FUNCTION f() RETURNS STRING LANGUAGE js AS 'return "\n";\n'` The definition_body is `return "\n";\n` Note that both \n are replaced with linebreaks.
	Sparkoptions SparkOptions `json:"sparkOptions,omitempty"` // Options for a user-defined Spark routine.
	Remotefunctionoptions RemoteFunctionOptions `json:"remoteFunctionOptions,omitempty"` // Options for a remote user-defined function.
	Creationtime string `json:"creationTime,omitempty"` // Output only. The time when this routine was created, in milliseconds since the epoch.
	Securitymode string `json:"securityMode,omitempty"` // Optional. The security mode of the routine, if defined. If not defined, the security mode is automatically determined from the routine's configuration.
	Returntype StandardSqlDataType `json:"returnType,omitempty"` // The data type of a variable such as a function argument. Examples include: * INT64: `{"typeKind": "INT64"}` * ARRAY: { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "STRING"} } * STRUCT>: { "typeKind": "STRUCT", "structType": { "fields": [ { "name": "x", "type": {"typeKind": "STRING"} }, { "name": "y", "type": { "typeKind": "ARRAY", "arrayElementType": {"typeKind": "DATE"} } } ] } }
	Arguments []Argument `json:"arguments,omitempty"` // Optional.
	Datagovernancetype string `json:"dataGovernanceType,omitempty"` // Optional. If set to `DATA_MASKING`, the function is validated and made available as a masking function. For more information, see [Create custom masking routines](https://cloud.google.com/bigquery/docs/user-defined-functions#custom-mask).
	Determinismlevel string `json:"determinismLevel,omitempty"` // Optional. The determinism level of the JavaScript UDF, if defined.
	Strictmode bool `json:"strictMode,omitempty"` // Optional. Use this option to catch many common errors. Error checking is not exhaustive, and successfully creating a procedure doesn't guarantee that the procedure will successfully execute at runtime. If `strictMode` is set to `TRUE`, the procedure body is further checked for errors such as non-existent tables or columns. The `CREATE PROCEDURE` statement fails if the body fails any of these checks. If `strictMode` is set to `FALSE`, the procedure body is checked only for syntax. For procedures that invoke themselves recursively, specify `strictMode=FALSE` to avoid non-existent procedure errors during validation. Default value is `TRUE`.
	Description string `json:"description,omitempty"` // Optional. The description of the routine, if defined.
	Routinereference RoutineReference `json:"routineReference,omitempty"` // Id path of a routine.
	Importedlibraries []string `json:"importedLibraries,omitempty"` // Optional. If language = "JAVASCRIPT", this field stores the path of the imported JAVASCRIPT libraries.
	Lastmodifiedtime string `json:"lastModifiedTime,omitempty"` // Output only. The time when this routine was last modified, in milliseconds since the epoch.
	Returntabletype StandardSqlTableType `json:"returnTableType,omitempty"` // A table type
	Routinetype string `json:"routineType,omitempty"` // Required. The type of routine.
}

// QueryResponse represents the QueryResponse schema from the OpenAPI specification
type QueryResponse struct {
	Schema TableSchema `json:"schema,omitempty"` // Schema of a table
	Jobreference JobReference `json:"jobReference,omitempty"` // A job reference is a fully qualified identifier for referring to a job.
	Cachehit bool `json:"cacheHit,omitempty"` // Whether the query result was fetched from the query cache.
	Jobcreationreason JobCreationReason `json:"jobCreationReason,omitempty"` // Reason about why a Job was created from a [`jobs.query`](https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query) method when used with `JOB_CREATION_OPTIONAL` Job creation mode. For [`jobs.insert`](https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/insert) method calls it will always be `REQUESTED`. This feature is not yet available. Jobs will always be created.
	Jobcomplete bool `json:"jobComplete,omitempty"` // Whether the query has completed or not. If rows or totalRows are present, this will always be true. If this is false, totalRows will not be available.
	Errors []ErrorProto `json:"errors,omitempty"` // Output only. The first errors or warnings encountered during the running of the job. The final message includes the number of errors that caused the process to stop. Errors here do not necessarily mean that the job has completed or was unsuccessful. For more information about error messages, see [Error messages](https://cloud.google.com/bigquery/docs/error-messages).
	Pagetoken string `json:"pageToken,omitempty"` // A token used for paging results. A non-empty token indicates that additional results are available. To see additional results, query the [`jobs.getQueryResults`](https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/getQueryResults) method. For more information, see [Paging through table data](https://cloud.google.com/bigquery/docs/paging-results).
	Kind string `json:"kind,omitempty"` // The resource type.
	Queryid string `json:"queryId,omitempty"` // Query ID for the completed query. This ID will be auto-generated. This field is not yet available and it is currently not guaranteed to be populated.
	Rows []TableRow `json:"rows,omitempty"` // An object with as many results as can be contained within the maximum permitted reply size. To get any additional rows, you can call GetQueryResults and specify the jobReference returned above.
	Totalbytesprocessed string `json:"totalBytesProcessed,omitempty"` // The total number of bytes processed for this query. If this query was a dry run, this is the number of bytes that would be processed if the query were run.
	Totalrows string `json:"totalRows,omitempty"` // The total number of rows in the complete query result set, which can be more than the number of rows in this single page of results.
	Numdmlaffectedrows string `json:"numDmlAffectedRows,omitempty"` // Output only. The number of rows affected by a DML statement. Present only for DML statements INSERT, UPDATE or DELETE.
	Sessioninfo SessionInfo `json:"sessionInfo,omitempty"` // [Preview] Information related to sessions.
	Dmlstats DmlStatistics `json:"dmlStats,omitempty"` // Detailed statistics for DML statements
}

// SystemVariables represents the SystemVariables schema from the OpenAPI specification
type SystemVariables struct {
	Types map[string]interface{} `json:"types,omitempty"` // Output only. Data type for each system variable.
	Values map[string]interface{} `json:"values,omitempty"` // Output only. Value for each system variable.
}

// TableDataList represents the TableDataList schema from the OpenAPI specification
type TableDataList struct {
	Etag string `json:"etag,omitempty"` // A hash of this page of results.
	Kind string `json:"kind,omitempty"` // The resource type of the response.
	Pagetoken string `json:"pageToken,omitempty"` // A token used for paging results. Providing this token instead of the startIndex parameter can help you retrieve stable results when an underlying table is changing.
	Rows []TableRow `json:"rows,omitempty"` // Rows of results.
	Totalrows string `json:"totalRows,omitempty"` // Total rows of the entire table. In order to show default value 0 we have to present it as string.
}

// TestIamPermissionsRequest represents the TestIamPermissionsRequest schema from the OpenAPI specification
type TestIamPermissionsRequest struct {
	Permissions []string `json:"permissions,omitempty"` // The set of permissions to check for the `resource`. Permissions with wildcards (such as `*` or `storage.*`) are not allowed. For more information see [IAM Overview](https://cloud.google.com/iam/docs/overview#permissions).
}
