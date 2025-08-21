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

func Bigquery_routines_updateHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		routineIdVal, ok := args["routineId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: routineId"), nil
		}
		routineId, ok := routineIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: routineId"), nil
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
		var requestBody models.Routine
		
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
		url := fmt.Sprintf("%s/projects/%s/datasets/%s/routines/%s%s", cfg.BaseURL, projectId, datasetId, routineId, queryString)
		req, err := http.NewRequest("PUT", url, bytes.NewBuffer(bodyBytes))
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
		var result models.Routine
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

func CreateBigquery_routines_updateTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("put_projects_projectId_datasets_datasetId_routines_routineId",
		mcp.WithDescription("Updates information in an existing routine. The update method replaces the entire Routine resource."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the routine to update")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset ID of the routine to update")),
		mcp.WithString("routineId", mcp.Required(), mcp.Description("Required. Routine ID of the routine to update")),
		mcp.WithArray("importedLibraries", mcp.Description("Input parameter: Optional. If language = \"JAVASCRIPT\", this field stores the path of the imported JAVASCRIPT libraries.")),
		mcp.WithString("lastModifiedTime", mcp.Description("Input parameter: Output only. The time when this routine was last modified, in milliseconds since the epoch.")),
		mcp.WithObject("returnTableType", mcp.Description("Input parameter: A table type")),
		mcp.WithString("routineType", mcp.Description("Input parameter: Required. The type of routine.")),
		mcp.WithString("etag", mcp.Description("Input parameter: Output only. A hash of this resource.")),
		mcp.WithString("language", mcp.Description("Input parameter: Optional. Defaults to \"SQL\" if remote_function_options field is absent, not set otherwise.")),
		mcp.WithString("definitionBody", mcp.Description("Input parameter: Required. The body of the routine. For functions, this is the expression in the AS clause. If language=SQL, it is the substring inside (but excluding) the parentheses. For example, for the function created with the following statement: `CREATE FUNCTION JoinLines(x string, y string) as (concat(x, \"\\n\", y))` The definition_body is `concat(x, \"\\n\", y)` (\\n is not replaced with linebreak). If language=JAVASCRIPT, it is the evaluated string in the AS clause. For example, for the function created with the following statement: `CREATE FUNCTION f() RETURNS STRING LANGUAGE js AS 'return \"\\n\";\\n'` The definition_body is `return \"\\n\";\\n` Note that both \\n are replaced with linebreaks.")),
		mcp.WithObject("sparkOptions", mcp.Description("Input parameter: Options for a user-defined Spark routine.")),
		mcp.WithObject("remoteFunctionOptions", mcp.Description("Input parameter: Options for a remote user-defined function.")),
		mcp.WithString("creationTime", mcp.Description("Input parameter: Output only. The time when this routine was created, in milliseconds since the epoch.")),
		mcp.WithString("securityMode", mcp.Description("Input parameter: Optional. The security mode of the routine, if defined. If not defined, the security mode is automatically determined from the routine's configuration.")),
		mcp.WithObject("returnType", mcp.Description("Input parameter: The data type of a variable such as a function argument. Examples include: * INT64: `{\"typeKind\": \"INT64\"}` * ARRAY: { \"typeKind\": \"ARRAY\", \"arrayElementType\": {\"typeKind\": \"STRING\"} } * STRUCT>: { \"typeKind\": \"STRUCT\", \"structType\": { \"fields\": [ { \"name\": \"x\", \"type\": {\"typeKind\": \"STRING\"} }, { \"name\": \"y\", \"type\": { \"typeKind\": \"ARRAY\", \"arrayElementType\": {\"typeKind\": \"DATE\"} } } ] } }")),
		mcp.WithArray("arguments", mcp.Description("Input parameter: Optional.")),
		mcp.WithString("dataGovernanceType", mcp.Description("Input parameter: Optional. If set to `DATA_MASKING`, the function is validated and made available as a masking function. For more information, see [Create custom masking routines](https://cloud.google.com/bigquery/docs/user-defined-functions#custom-mask).")),
		mcp.WithString("determinismLevel", mcp.Description("Input parameter: Optional. The determinism level of the JavaScript UDF, if defined.")),
		mcp.WithBoolean("strictMode", mcp.Description("Input parameter: Optional. Use this option to catch many common errors. Error checking is not exhaustive, and successfully creating a procedure doesn't guarantee that the procedure will successfully execute at runtime. If `strictMode` is set to `TRUE`, the procedure body is further checked for errors such as non-existent tables or columns. The `CREATE PROCEDURE` statement fails if the body fails any of these checks. If `strictMode` is set to `FALSE`, the procedure body is checked only for syntax. For procedures that invoke themselves recursively, specify `strictMode=FALSE` to avoid non-existent procedure errors during validation. Default value is `TRUE`.")),
		mcp.WithString("description", mcp.Description("Input parameter: Optional. The description of the routine, if defined.")),
		mcp.WithObject("routineReference", mcp.Description("Input parameter: Id path of a routine.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_routines_updateHandler(cfg),
	}
}
