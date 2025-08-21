package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"bytes"

	"github.com/bigquery-api/mcp-server/config"
	"github.com/bigquery-api/mcp-server/models"
	"github.com/mark3labs/mcp-go/mcp"
)

func Bigquery_models_patchHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		args, ok := request.Params.Arguments.(map[string]any)
		if !ok {
			return mcp.NewToolResultError("Invalid arguments object"), nil
		}
		projectIdVal, ok := args["projectId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: projectId"), nil
		}
		projectId, ok := projectIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: projectId"), nil
		}
		datasetIdVal, ok := args["datasetId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: datasetId"), nil
		}
		datasetId, ok := datasetIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: datasetId"), nil
		}
		modelIdVal, ok := args["modelId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: modelId"), nil
		}
		modelId, ok := modelIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: modelId"), nil
		}
		queryParams := make([]string, 0)
		// Handle multiple authentication parameters
		if cfg.BearerToken != "" {
			queryParams = append(queryParams, fmt.Sprintf("access_token=%s", cfg.BearerToken))
		}
		if cfg.APIKey != "" {
			queryParams = append(queryParams, fmt.Sprintf("key=%s", cfg.APIKey))
		}
		if cfg.BearerToken != "" {
			queryParams = append(queryParams, fmt.Sprintf("oauth_token=%s", cfg.BearerToken))
		}
		queryString := ""
		if len(queryParams) > 0 {
			queryString = "?" + strings.Join(queryParams, "&")
		}
		// Create properly typed request body using the generated schema
		var requestBody models.Model
		
		// Optimized: Single marshal/unmarshal with JSON tags handling field mapping
		if argsJSON, err := json.Marshal(args); err == nil {
			if err := json.Unmarshal(argsJSON, &requestBody); err != nil {
				return mcp.NewToolResultError(fmt.Sprintf("Failed to convert arguments to request type: %v", err)), nil
			}
		} else {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to marshal arguments: %v", err)), nil
		}
		
		bodyBytes, err := json.Marshal(requestBody)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to encode request body", err), nil
		}
		url := fmt.Sprintf("%s/projects/%s/datasets/%s/models/%s%s", cfg.BaseURL, projectId, datasetId, modelId, queryString)
		req, err := http.NewRequest("PATCH", url, bytes.NewBuffer(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to create request", err), nil
		}
		// Set authentication based on auth type
		// Handle multiple authentication parameters
		// API key already added to query string
		// API key already added to query string
		// API key already added to query string
		req.Header.Set("Accept", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Request failed", err), nil
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to read response body", err), nil
		}

		if resp.StatusCode >= 400 {
			return mcp.NewToolResultError(fmt.Sprintf("API error: %s", body)), nil
		}
		// Use properly typed response
		var result models.Model
		if err := json.Unmarshal(body, &result); err != nil {
			// Fallback to raw text if unmarshaling fails
			return mcp.NewToolResultText(string(body)), nil
		}

		prettyJSON, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to format JSON", err), nil
		}

		return mcp.NewToolResultText(string(prettyJSON)), nil
	}
}

func CreateBigquery_models_patchTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("patch_projects_projectId_datasets_datasetId_models_modelId",
		mcp.WithDescription("Patch specific fields in the specified model."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the model to patch.")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset ID of the model to patch.")),
		mcp.WithString("modelId", mcp.Required(), mcp.Description("Required. Model ID of the model to patch.")),
		mcp.WithObject("encryptionConfiguration", mcp.Description("")),
		mcp.WithArray("trainingRuns", mcp.Description("Input parameter: Information for all training runs in increasing order of start_time.")),
		mcp.WithString("etag", mcp.Description("Input parameter: Output only. A hash of this resource.")),
		mcp.WithString("friendlyName", mcp.Description("Input parameter: Optional. A descriptive name for this model.")),
		mcp.WithString("bestTrialId", mcp.Description("Input parameter: The best trial_id across all training runs.")),
		mcp.WithArray("transformColumns", mcp.Description("Input parameter: Output only. This field will be populated if a TRANSFORM clause was used to train a model. TRANSFORM clause (if used) takes feature_columns as input and outputs transform_columns. transform_columns then are used to train the model.")),
		mcp.WithArray("optimalTrialIds", mcp.Description("Input parameter: Output only. For single-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, it only contains the best trial. For multi-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, it contains all Pareto optimal trials sorted by trial_id.")),
		mcp.WithObject("hparamSearchSpaces", mcp.Description("Input parameter: Hyperparameter search spaces. These should be a subset of training_options.")),
		mcp.WithObject("labels", mcp.Description("Input parameter: The labels associated with this model. You can use these to organize and group your models. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label values are optional. Label keys must start with a letter and each label in the list must have a different key.")),
		mcp.WithString("expirationTime", mcp.Description("Input parameter: Optional. The time when this model expires, in milliseconds since the epoch. If not present, the model will persist indefinitely. Expired models will be deleted and their storage reclaimed. The defaultTableExpirationMs property of the encapsulating dataset can be used to set a default expirationTime on newly created models.")),
		mcp.WithArray("featureColumns", mcp.Description("Input parameter: Output only. Input feature columns for the model inference. If the model is trained with TRANSFORM clause, these are the input of the TRANSFORM clause.")),
		mcp.WithString("location", mcp.Description("Input parameter: Output only. The geographic location where the model resides. This value is inherited from the dataset.")),
		mcp.WithObject("modelReference", mcp.Description("Input parameter: Id path of a model.")),
		mcp.WithString("modelType", mcp.Description("Input parameter: Output only. Type of the model resource.")),
		mcp.WithObject("remoteModelInfo", mcp.Description("Input parameter: Remote Model Info")),
		mcp.WithArray("hparamTrials", mcp.Description("Input parameter: Output only. Trials of a [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) model sorted by trial_id.")),
		mcp.WithArray("labelColumns", mcp.Description("Input parameter: Output only. Label columns that were used to train this model. The output of the model will have a \"predicted_\" prefix to these columns.")),
		mcp.WithString("creationTime", mcp.Description("Input parameter: Output only. The time when this model was created, in millisecs since the epoch.")),
		mcp.WithString("description", mcp.Description("Input parameter: Optional. A user-friendly description of this model.")),
		mcp.WithString("lastModifiedTime", mcp.Description("Input parameter: Output only. The time when this model was last modified, in millisecs since the epoch.")),
		mcp.WithString("defaultTrialId", mcp.Description("Input parameter: Output only. The default trial_id to use in TVFs when the trial_id is not passed in. For single-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, this is the best trial ID. For multi-objective [hyperparameter tuning](/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models, this is the smallest trial ID among all Pareto optimal trials.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_models_patchHandler(cfg),
	}
}
