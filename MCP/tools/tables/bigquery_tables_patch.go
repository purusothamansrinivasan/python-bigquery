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

func Bigquery_tables_patchHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		tableIdVal, ok := args["tableId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: tableId"), nil
		}
		tableId, ok := tableIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: tableId"), nil
		}
		queryParams := make([]string, 0)
		if val, ok := args["autodetect_schema"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("autodetect_schema=%v", val))
		}
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
		var requestBody models.Table
		
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
		url := fmt.Sprintf("%s/projects/%s/datasets/%s/tables/%s%s", cfg.BaseURL, projectId, datasetId, tableId, queryString)
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
		var result models.Table
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

func CreateBigquery_tables_patchTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("patch_projects_projectId_datasets_datasetId_tables_tableId",
		mcp.WithDescription("Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource. This method supports RFC5789 patch semantics."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the table to update")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset ID of the table to update")),
		mcp.WithString("tableId", mcp.Required(), mcp.Description("Required. Table ID of the table to update")),
		mcp.WithBoolean("autodetect_schema", mcp.Description("Optional.  When true will autodetect schema, else will keep original schema")),
		mcp.WithString("kind", mcp.Description("Input parameter: The type of resource ID.")),
		mcp.WithString("numLongTermLogicalBytes", mcp.Description("Input parameter: Output only. Number of logical bytes that are more than 90 days old.")),
		mcp.WithObject("rangePartitioning", mcp.Description("")),
		mcp.WithObject("timePartitioning", mcp.Description("")),
		mcp.WithObject("cloneDefinition", mcp.Description("Input parameter: Information about base table and clone time of a table clone.")),
		mcp.WithObject("materializedViewStatus", mcp.Description("Input parameter: Status of a materialized view. The last refresh timestamp status is omitted here, but is present in the MaterializedViewDefinition message.")),
		mcp.WithString("numLongTermBytes", mcp.Description("Input parameter: Output only. The number of logical bytes in the table that are considered \"long-term storage\".")),
		mcp.WithString("numLongTermPhysicalBytes", mcp.Description("Input parameter: Output only. Number of physical bytes more than 90 days old. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.")),
		mcp.WithBoolean("requirePartitionFilter", mcp.Description("Input parameter: Optional. If set to true, queries over this table require a partition filter that can be used for partition elimination to be specified.")),
		mcp.WithObject("labels", mcp.Description("Input parameter: The labels associated with this table. You can use these to organize and group your tables. Label keys and values can be no longer than 63 characters, can only contain lowercase letters, numeric characters, underscores and dashes. International characters are allowed. Label values are optional. Label keys must start with a letter and each label in the list must have a different key.")),
		mcp.WithString("numActivePhysicalBytes", mcp.Description("Input parameter: Output only. Number of physical bytes less than 90 days old. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.")),
		mcp.WithObject("clustering", mcp.Description("Input parameter: Configures table clustering.")),
		mcp.WithString("id", mcp.Description("Input parameter: Output only. An opaque ID uniquely identifying the table.")),
		mcp.WithObject("biglakeConfiguration", mcp.Description("Input parameter: Configuration for BigLake managed tables.")),
		mcp.WithString("numTimeTravelPhysicalBytes", mcp.Description("Input parameter: Output only. Number of physical bytes used by time travel storage (deleted or changed data). This data is not kept in real time, and might be delayed by a few seconds to a few minutes.")),
		mcp.WithObject("resourceTags", mcp.Description("Input parameter: [Optional] The tags associated with this table. Tag keys are globally unique. See additional information on [tags](https://cloud.google.com/iam/docs/tags-access-control#definitions). An object containing a list of \"key\": value pairs. The key is the namespaced friendly name of the tag key, e.g. \"12345/environment\" where 12345 is parent id. The value is the friendly short name of the tag value, e.g. \"production\".")),
		mcp.WithObject("encryptionConfiguration", mcp.Description("")),
		mcp.WithString("lastModifiedTime", mcp.Description("Input parameter: Output only. The time when this table was last modified, in milliseconds since the epoch.")),
		mcp.WithObject("view", mcp.Description("Input parameter: Describes the definition of a logical view.")),
		mcp.WithObject("model", mcp.Description("")),
		mcp.WithString("location", mcp.Description("Input parameter: Output only. The geographic location where the table resides. This value is inherited from the dataset.")),
		mcp.WithString("numTotalLogicalBytes", mcp.Description("Input parameter: Output only. Total number of logical bytes in the table or materialized view.")),
		mcp.WithString("numActiveLogicalBytes", mcp.Description("Input parameter: Output only. Number of logical bytes that are less than 90 days old.")),
		mcp.WithString("numPartitions", mcp.Description("Input parameter: Output only. The number of partitions present in the table or materialized view. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.")),
		mcp.WithString("numRows", mcp.Description("Input parameter: Output only. The number of rows of data in this table, excluding any data in the streaming buffer.")),
		mcp.WithObject("tableConstraints", mcp.Description("Input parameter: The TableConstraints defines the primary key and foreign key.")),
		mcp.WithObject("streamingBuffer", mcp.Description("")),
		mcp.WithString("defaultCollation", mcp.Description("Input parameter: Optional. Defines the default collation specification of new STRING fields in the table. During table creation or update, if a STRING field is added to this table without explicit collation specified, then the table inherits the table default collation. A change to this field affects only fields added afterwards, and does not alter the existing fields. The following values are supported: * 'und:ci': undetermined locale, case insensitive. * '': empty string. Default to case-sensitive behavior.")),
		mcp.WithObject("tableReplicationInfo", mcp.Description("Input parameter: Replication info of a table created using `AS REPLICA` DDL like: `CREATE MATERIALIZED VIEW mv1 AS REPLICA OF src_mv`")),
		mcp.WithString("numPhysicalBytes", mcp.Description("Input parameter: Output only. The physical size of this table in bytes. This includes storage used for time travel.")),
		mcp.WithString("selfLink", mcp.Description("Input parameter: Output only. A URL that can be used to access this resource again.")),
		mcp.WithString("defaultRoundingMode", mcp.Description("Input parameter: Optional. Defines the default rounding mode specification of new decimal fields (NUMERIC OR BIGNUMERIC) in the table. During table creation or update, if a decimal field is added to this table without an explicit rounding mode specified, then the field inherits the table default rounding mode. Changing this field doesn't affect existing fields.")),
		mcp.WithObject("schema", mcp.Description("Input parameter: Schema of a table")),
		mcp.WithObject("snapshotDefinition", mcp.Description("Input parameter: Information about base table and snapshot time of the snapshot.")),
		mcp.WithString("friendlyName", mcp.Description("Input parameter: Optional. A descriptive name for this table.")),
		mcp.WithString("type", mcp.Description("Input parameter: Output only. Describes the table type. The following values are supported: * `TABLE`: A normal BigQuery table. * `VIEW`: A virtual table defined by a SQL query. * `EXTERNAL`: A table that references data stored in an external storage system, such as Google Cloud Storage. * `MATERIALIZED_VIEW`: A precomputed view defined by a SQL query. * `SNAPSHOT`: An immutable BigQuery table that preserves the contents of a base table at a particular time. See additional information on [table snapshots](/bigquery/docs/table-snapshots-intro). The default value is `TABLE`.")),
		mcp.WithString("description", mcp.Description("Input parameter: Optional. A user-friendly description of this table.")),
		mcp.WithObject("tableReference", mcp.Description("")),
		mcp.WithString("creationTime", mcp.Description("Input parameter: Output only. The time when this table was created, in milliseconds since the epoch.")),
		mcp.WithString("etag", mcp.Description("Input parameter: Output only. A hash of this resource.")),
		mcp.WithString("numBytes", mcp.Description("Input parameter: Output only. The size of this table in logical bytes, excluding any data in the streaming buffer.")),
		mcp.WithString("maxStaleness", mcp.Description("Input parameter: Optional. The maximum staleness of data that could be returned when the table (or stale MV) is queried. Staleness encoded as a string encoding of sql IntervalValue type.")),
		mcp.WithString("numTotalPhysicalBytes", mcp.Description("Input parameter: Output only. The physical size of this table in bytes. This also includes storage used for time travel. This data is not kept in real time, and might be delayed by a few seconds to a few minutes.")),
		mcp.WithObject("externalDataConfiguration", mcp.Description("")),
		mcp.WithObject("materializedView", mcp.Description("Input parameter: Definition and configuration of a materialized view.")),
		mcp.WithString("expirationTime", mcp.Description("Input parameter: Optional. The time when this table expires, in milliseconds since the epoch. If not present, the table will persist indefinitely. Expired tables will be deleted and their storage reclaimed. The defaultTableExpirationMs property of the encapsulating dataset can be used to set a default expirationTime on newly created tables.")),
		mcp.WithArray("replicas", mcp.Description("Input parameter: Optional. Output only. Table references of all replicas currently active on the table.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_tables_patchHandler(cfg),
	}
}
