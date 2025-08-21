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

func Bigquery_datasets_patchHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		var requestBody models.Dataset
		
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
		url := fmt.Sprintf("%s/projects/%s/datasets/%s%s", cfg.BaseURL, projectId, datasetId, queryString)
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
		var result models.Dataset
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

func CreateBigquery_datasets_patchTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("patch_projects_projectId_datasets_datasetId",
		mcp.WithDescription("Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource. This method supports RFC5789 patch semantics."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the dataset being updated")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset ID of the dataset being updated")),
		mcp.WithObject("datasetReference", mcp.Description("")),
		mcp.WithBoolean("satisfiesPzi", mcp.Description("Input parameter: Output only. Reserved for future use.")),
		mcp.WithString("type", mcp.Description("Input parameter: Output only. Same as `type` in `ListFormatDataset`. The type of the dataset, one of: * DEFAULT - only accessible by owner and authorized accounts, * PUBLIC - accessible by everyone, * LINKED - linked dataset, * EXTERNAL - dataset with definition in external metadata catalog. -- *BIGLAKE_METASTORE - dataset that references a database created in BigLakeMetastore service. --")),
		mcp.WithString("id", mcp.Description("Input parameter: Output only. The fully-qualified unique name of the dataset in the format projectId:datasetId. The dataset name without the project name is given in the datasetId field. When creating a new dataset, leave this field blank, and instead specify the datasetId field.")),
		mcp.WithString("creationTime", mcp.Description("Input parameter: Output only. The time when this dataset was created, in milliseconds since the epoch.")),
		mcp.WithString("kind", mcp.Description("Input parameter: Output only. The resource type.")),
		mcp.WithString("lastModifiedTime", mcp.Description("Input parameter: Output only. The date when this dataset was last modified, in milliseconds since the epoch.")),
		mcp.WithString("defaultTableExpirationMs", mcp.Description("Input parameter: Optional. The default lifetime of all tables in the dataset, in milliseconds. The minimum lifetime value is 3600000 milliseconds (one hour). To clear an existing default expiration with a PATCH request, set to 0. Once this property is set, all newly-created tables in the dataset will have an expirationTime property set to the creation time plus the value in this property, and changing the value will only affect new tables, not existing ones. When the expirationTime for a given table is reached, that table will be deleted automatically. If a table's expirationTime is modified or removed before the table expires, or if you provide an explicit expirationTime when creating a table, that value takes precedence over the default expiration time indicated by this property.")),
		mcp.WithArray("tags", mcp.Description("Input parameter: Output only. Tags for the Dataset.")),
		mcp.WithString("defaultPartitionExpirationMs", mcp.Description("Input parameter: This default partition expiration, expressed in milliseconds. When new time-partitioned tables are created in a dataset where this property is set, the table will inherit this value, propagated as the `TimePartitioning.expirationMs` property on the new table. If you set `TimePartitioning.expirationMs` explicitly when creating a table, the `defaultPartitionExpirationMs` of the containing dataset is ignored. When creating a partitioned table, if `defaultPartitionExpirationMs` is set, the `defaultTableExpirationMs` value is ignored and the table will not be inherit a table expiration deadline.")),
		mcp.WithString("friendlyName", mcp.Description("Input parameter: Optional. A descriptive name for the dataset.")),
		mcp.WithString("maxTimeTravelHours", mcp.Description("Input parameter: Optional. Defines the time travel window in hours. The value can be from 48 to 168 hours (2 to 7 days). The default value is 168 hours if this is not set.")),
		mcp.WithString("defaultRoundingMode", mcp.Description("Input parameter: Optional. Defines the default rounding mode specification of new tables created within this dataset. During table creation, if this field is specified, the table within this dataset will inherit the default rounding mode of the dataset. Setting the default rounding mode on a table overrides this option. Existing tables in the dataset are unaffected. If columns are defined during that table creation, they will immediately inherit the table's default rounding mode, unless otherwise specified.")),
		mcp.WithObject("labels", mcp.Description("Input parameter: The labels associated with this dataset. You can use these to organize and group your datasets. You can set this property when inserting or updating a dataset. See Creating and Updating Dataset Labels for more information.")),
		mcp.WithBoolean("isCaseInsensitive", mcp.Description("Input parameter: Optional. TRUE if the dataset and its table names are case-insensitive, otherwise FALSE. By default, this is FALSE, which means the dataset and its table names are case-sensitive. This field does not affect routine references.")),
		mcp.WithBoolean("satisfiesPzs", mcp.Description("Input parameter: Output only. Reserved for future use.")),
		mcp.WithString("storageBillingModel", mcp.Description("Input parameter: Optional. Updates storage_billing_model for the dataset.")),
		mcp.WithString("description", mcp.Description("Input parameter: Optional. A user-friendly description of the dataset.")),
		mcp.WithString("etag", mcp.Description("Input parameter: Output only. A hash of the resource.")),
		mcp.WithString("defaultCollation", mcp.Description("Input parameter: Optional. Defines the default collation specification of future tables created in the dataset. If a table is created in this dataset without table-level default collation, then the table inherits the dataset default collation, which is applied to the string fields that do not have explicit collation specified. A change to this field affects only tables created afterwards, and does not alter the existing tables. The following values are supported: * 'und:ci': undetermined locale, case insensitive. * '': empty string. Default to case-sensitive behavior.")),
		mcp.WithArray("access", mcp.Description("Input parameter: Optional. An array of objects that define dataset access for one or more entities. You can set this property when inserting or updating a dataset in order to control who is allowed to access the data. If unspecified at dataset creation time, BigQuery adds default dataset access for the following entities: access.specialGroup: projectReaders; access.role: READER; access.specialGroup: projectWriters; access.role: WRITER; access.specialGroup: projectOwners; access.role: OWNER; access.userByEmail: [dataset creator email]; access.role: OWNER;")),
		mcp.WithObject("externalDatasetReference", mcp.Description("Input parameter: Configures the access a dataset defined in an external metadata storage.")),
		mcp.WithString("location", mcp.Description("Input parameter: The geographic location where the dataset should reside. See https://cloud.google.com/bigquery/docs/locations for supported locations.")),
		mcp.WithString("selfLink", mcp.Description("Input parameter: Output only. A URL that can be used to access the resource again. You can use this URL in Get or Update requests to the resource.")),
		mcp.WithObject("defaultEncryptionConfiguration", mcp.Description("")),
		mcp.WithObject("linkedDatasetSource", mcp.Description("Input parameter: A dataset source type which refers to another BigQuery dataset.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_datasets_patchHandler(cfg),
	}
}
