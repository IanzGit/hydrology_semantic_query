# 水害场景语义模型人工治理指南

## 1. 文档定位

本文用于人工治理 `hydrology_semantic_query` 的高频业务 View、长尾基础 Cube、Join Graph 与语义样例，不属于通用建模 skill，也不应被自动生成器当作数据库关系证据。

本次治理参考：

- `docs/水害场景设计20260508-场景1.docx` 中“1. 监测设备情况”“2. 报警情况”“3. 预警情况”的说明、关系图和查询片段。
- MySQL 中相关表的字段、主键、唯一性、关联匹配和当前数据量。
- 当前部署的 Cube 版本及其 View、Join、成员别名和访问策略能力。

文档中的查询片段只用于识别业务入口、投影字段、关联方向和固定条件。模型不复制完整查询语句，也不把 `SELECT *` 解释为公开全部字段。

## 2. 当前公开范围

公开目录保留三个高频业务 View：

| View | 根粒度 | 业务用途 | 当前数据快照基线 |
| --- | --- | --- | ---: |
| `hydrology_monitoring_devices` | 一个传感器 | 查询启用设备下的启用传感器及状态，可选标签筛选 | 384 |
| `hydrology_single_factor_alarms` | 一条单因素报警事件 | 查询启用报警及其启用传感器、设备 | 1198 |
| `hydrology_multifactor_warnings` | 一条多因素预警事件 | 查询启用多因素预警及其启用配置 | 84 |

基线是 2026-08-14 数据快照的对照值，只用于发现模型改动造成的丢行或扇出，不是长期业务常量。数据库数据变化后，应重新计算并更新验收记录。

以下能力不作为独立高频业务 View：

- 完整标签目录和父子标签树。
- 传感器与标签关系诊断。
- 多因素预警配置管理清单。
- 多因素配置的逐项指标明细。
- 单独的设备主数据或传感器主数据目录。

这些能力通过受治理基础 Cube 支持长尾组合查询，但不会恢复为宽口径业务 View。`role_label_parent` 没有进入当前精确 Join Graph，继续保持私有。

### 2.1 公开基础 Cube

七个 `base_*` Cube 只公开稳定业务成员，并承担以下根粒度：

| Cube | 根粒度 | 长尾用途 |
| --- | --- | --- |
| `base_device_info` | 一台设备 | 设备主数据与状态 |
| `base_device_x_value` | 一个传感器 | 传感器主数据、状态与设备关联 |
| `base_label` | 一个标签 | 标签字典，不包含父子树关联 |
| `base_label_sensor` | 一条唯一传感器标签关系 | 标签关系与源记录数核验 |
| `base_multifactor_sensor` | 一个多因素指标 | 配置指标与当前值 |
| `base_warn_state_info` | 一条报警或预警事件 | 未施加业务类型固定条件的原始事件分析 |
| `base_water_warn_sensor_set` | 一个多因素预警配置 | 配置、风险等级和当前状态 |

基础 Cube 不继承三个 View 的强制业务过滤。例如 `base_warn_state_info` 同时包含不同来源类型，调用方必须显式过滤；需要固定口径时优先命中对应 View。寄存器地址、通信解析字段、脚本内容、脚本或历史文件路径、函数参数和审计账号等技术成员保持隐藏。

### 2.2 精确 Join Graph

精确边由 Cube YAML 的 `meta.join_edges` 维护，当前图为：

- `base_device_x_value` — `base_device_info`
- `base_device_x_value` — `base_label_sensor` — `base_label`
- `base_warn_state_info` — `base_device_x_value`
- `base_warn_state_info` — `base_water_warn_sensor_set`
- `base_multifactor_sensor` — `base_device_x_value`
- `base_multifactor_sensor` — `base_water_warn_sensor_set`

`connectedComponent` 和业务域只用于候选扩展，不证明精确可执行关系。Cube 查询必须选择最多四个模型，包含最小路径上的中间 Cube，并通过验证器证明连通。如果一对模型存在多条等长最短路径，当前模型不能证明正确业务路径，必须在调用 `/load` 前返回 `semantic_model_gap`，不能让 Cube 或查询生成模型猜测。

## 3. 证据与治理边界

结构层和业务层分别判断证据：

- 主键、Join 方向、关系基数、孤儿和扇出以数据库约束及实测结果为准。文档关系图只能提出候选关系，不能覆盖不成立的数据库关系。
- 固定过滤、公开字段、业务命名和结果用途以业务人员确认的定义为准，文档正文和查询片段用于补充说明及验收。
- 仅有字段同名、自然语言推测或查询书写顺序的关系不进入公开 View。

文档和数据库冲突时不要折中猜测。先保留数据库可验证的结构，再把业务口径冲突登记为待确认项。查询片段可以作为验收对照，但不能替代基础表关系验证。

### 3.1 Cube 1.6.70 的强制过滤实现

当前部署使用 Cube 1.6.70 和经典 REST 查询规划器。实测该版本会接受并编译 View 的 `default_filters` 字段，但经典规划器执行 `/load` 时不会应用这些过滤，因此不能把它作为当前环境的业务正确性保障。

三个业务 View 使用 `role: "*"` 的访问策略，并在 `row_level.filters` 中声明不可取消的业务条件；`member_level.includes: "*"` 用于满足 1.6 系列的严格策略成员匹配。当前 MySQL 启用字段以 0/1 保存，过滤值使用字符串 `"1"`；实测写成 `"true"` 会被 MySQL 按字符串比较并得到错误结果。升级 Cube 或切换查询规划器后，必须重新验证实际生成查询和结果计数，再决定是否迁移回 `default_filters`。

## 4. 监测设备情况

### 4.1 粒度和关系

根 Cube 是 `base_device_x_value`，一行代表一个传感器。

关系路径：

- `base_device_x_value` 到 `base_device_info`：多对一，用于取得设备名称和设备启用状态。
- `base_device_x_value` 到 `base_label_sensor`：一对多，仅在需要标签筛选或标签成员时激活。

`base_label_sensor` 对物理表中的相同传感器、标签组合先做唯一关系治理，并保留源记录数用于内部核验。该治理避免源表的完全重复关系直接放大结果，但不会改变“一个传感器可以有多个标签”的事实。

### 4.2 固定业务条件

View 使用通配角色访问策略中的不可省略行过滤：

- 传感器启用。
- 所属设备启用。

Cube 的 Join 为左连接语义；对关联设备的启用字段应用固定过滤后，最终结果与文档中的有效设备内连接口径一致。没有有效启用设备的传感器不会进入该业务 View。

### 4.3 公开字段

当前只公开高频查询需要的稳定字段：

- `sensor_id`
- `device_id`
- `device_name`
- `sensor_name`
- `sensor_state`
- `sensor_state_updated_at`
- `label_id`
- `sensor_is_enabled`
- `device_is_enabled`
- `monitoring_sensor_count`

不要因为物理表存在寄存器、阈值、解析规则或通信配置字段，就自动将它们全部加入该 View。新增字段前应先确认它是否属于“监测设备情况”高频结果。

### 4.4 标签筛选口径

文档中的标签 ID 列表按“命中任意一个标签”解释，不是“同时具备全部标签”。

需要特别注意：

- 不选择也不过滤 `label_id` 时，查询保持传感器粒度。
- 选择 `label_id` 后，结果粒度变为“传感器—标签关系”。
- 同时过滤多个标签时，一个传感器命中多个标签可能返回多行；这与文档关联查询的自然语义一致。
- 聚合 `monitoring_sensor_count` 依赖根传感器主键进行防扇出，但无分组明细查询不会自动替代业务去重。

当前快照选择关联传感器最多的两个标签时，明细返回 537 行、涉及 283 个不同传感器，而 `monitoring_sensor_count` 正确返回 283。该差异是多标签关系展开，不是物理重复关系重新进入模型。

如果业务要求“多标签任意命中，但每个传感器只返回一行”，普通 View 的动态 Join 不能稳定表达该半连接口径。应由业务人员明确授权后，单独治理基于 `EXISTS` 或等价集合语义的业务 Cube，而不是在当前 View 中静默去重。

完整标签树属于筛选器字典，不由三个公开业务 View 提供。前端需要标签选择器时，应调用独立字典或配置管理接口。

## 5. 单因素报警情况

### 5.1 粒度和关系

根 Cube 是 `base_warn_state_info`，一行代表一条报警事件。

关系路径：

- 报警来源到 `base_device_x_value`：多对一，并限定来源类型为单因素传感器。
- 传感器到 `base_device_info`：多对一。

### 5.2 固定业务条件

View 使用通配角色访问策略中的不可省略行过滤：

- `source_type = 1`。
- 报警记录启用。
- 来源传感器启用。
- 所属设备启用。

这组条件补全了文档关系图和章节语义中已经表达、但查询片段没有完整写出的来源类型边界，防止不同来源类型恰好出现相同 ID 时被误关联。

### 5.3 公开字段

- `alarm_event_id`
- `sensor_id`
- `device_id`
- `device_name`
- `sensor_name`
- `alarm_name`
- `alarm_value`
- `current_level`
- `highest_level`
- `started_at`
- `source_type`
- 三方启用状态
- `alarm_event_count`

`highest_level` 的物理字段是历史最高报警等级，不能命名或解释为当前等级。当前等级使用 `current_level`。

文档没有定义默认时间窗口、排序、分页、误报排除或已解除排除条件，因此 View 不补这些条件。它们应在具体查询中显式提供，或在业务确认后另行治理。

### 5.4 待人工确认

文档对来源类型的文字说明存在不一致。当前模型依据表关系、关系图及现有数据，将 `source_type = 1` 作为单因素传感器报警，将 `source_type = 3` 作为多因素预警。业务人员应最终确认该枚举；确认结果变化时，需要同时修改基础 Join、View 强制行过滤、描述、验收基线和回归测试。

## 6. 多因素预警情况

### 6.1 粒度和关系

根 Cube 是 `base_warn_state_info`，一行代表一条多因素预警事件。

预警事件通过来源 ID 多对一关联 `base_water_warn_sensor_set`。该关系同时得到文档关系图、查询片段和数据库匹配结果支持。

### 6.2 固定业务条件

View 使用通配角色访问策略中的不可省略行过滤：

- `source_type = 3`。
- 预警事件启用。
- 关联预警配置启用。

没有有效启用配置的事件不会进入该业务 View。

### 6.3 公开字段

事件侧公开：

- `warning_event_id`
- `configuration_id`
- `warning_name`
- `warning_value`
- `current_level`
- `highest_level`
- `started_at`
- `warning_place`
- `station_name`
- `source_type`
- `warning_is_enabled`
- `warning_event_count`

配置侧只公开查询所需的稳定上下文：

- `configuration_name`
- `is_custom`
- `configuration_category`
- `configuration_place`
- `risk_level`
- `configuration_current_value`
- `configuration_value_updated_at`
- `warning_source_data`
- `strategy_type`
- `warning_value_categories`
- `unit`
- `configuration_is_enabled`

脚本正文、脚本路径、算法参数、审计字段及其他配置实现细节不应因文档使用了全字段投影就自动公开。

### 6.4 配置和指标边界

文档还包含“全部启用多因素配置”的辅助查询。多因素 View 选择事件粒度；尚未产生事件的启用配置不会出现在这里。完整配置清单优先由配置管理接口提供，受控分析也可以使用 `base_water_warn_sensor_set` 并显式过滤启用状态。

`multifactor_sensor.parentId` 可以关联预警配置，但逐项指标是一对多关系，会把一条预警事件展开为多行。文档的实际预警查询没有使用该表，且正文对指标传感器与预警来源的描述存在冲突，因此指标不进入当前公开 View。需要“每条事件对应哪些指标”时，可以使用公开基础 Cube 探索，但必须显式承认当前配置指标的一对多粒度；如果要把它解释为事件发生时的指标快照，仍需先确认时间语义。

## 7. 检索、路由和样例治理

查询理解先拆出指标、维度、时间、过滤、排序和条数概念。View、Cube、全局 Member 和 Example 使用独立索引与独立 Top-K；Member 命中可以反查并拉回父 Cube。重排固定使用语义相关度 40%、概念覆盖率 30%、精确词或别名 10%、图连通性 10%、业务优先级 10%。

只有单个 View 对全部必需概念达到 100% 覆盖时才能走 View 快路径。其他请求进入 Cube Graph，提示词上下文只包含选中的模型、6 至 12 个公开成员、最小 Join Path、固定语义和允许成员，不包含完整原始目录。向量不可用时改用同样有界的词法检索；结果为空或可修正失败时按连通分量或业务域分批分析；无法形成连通、无歧义模型时返回结构化语义缺口。

## 8. 从业务查询片段治理 View 的方法

不要把完整查询语句粘贴进 Cube。按以下映射逐项治理：

| 查询片段中的信息 | Cube 中的治理位置 |
| --- | --- |
| 主表 | 选择根 Cube 和结果粒度 |
| 关联条件 | 基础 Cube 的 `joins` 和 View 的 `join_path` |
| 稳定的业务必选条件 | View 的 `access_policy.row_level.filters` |
| 返回字段 | View 的显式 `includes` 和业务别名 |
| 标签 ID 列表 | 对 `label_id` 的查询级过滤 |
| 时间范围、排序、分页 | 具体 SemanticQuery，不从文档空白处推断 |
| 全字段投影 | 人工挑选稳定公开字段，不直接使用 `includes: "*"` |

每次治理前先写清三个问题：

1. 结果的一行代表什么。
2. 哪些条件是所有调用者都不能取消的业务边界。
3. 哪些关联会改变根粒度或产生一对多展开。

## 9. 旧 View 迁移

| 原公开 View | 处理方式 |
| --- | --- |
| `hydrology_devices` | 不再单独公开；设备字段按需进入监测设备 View |
| `hydrology_sensors` | 由 `hydrology_monitoring_devices` 取代 |
| `hydrology_labels` | 删除公开入口；完整标签树转字典接口 |
| `hydrology_sensor_labels` | 删除公开入口；标签 ID 作为监测设备的可选筛选字段 |
| `hydrology_alarm_events` | 由 `hydrology_single_factor_alarms` 取代并固化单因素边界 |
| `hydrology_warning_events` | 由 `hydrology_multifactor_warnings` 取代并固化多因素边界 |
| `hydrology_warning_configurations` | 删除公开入口；完整清单转配置管理接口 |
| `hydrology_multifactor_indicators` | 删除公开入口；等待独立业务口径治理 |

这是公开 API 的名称和成员迁移。调用方、缓存向量、Benchmark 和固定查询样例中如仍引用旧名称，应逐项迁移，不能通过恢复多余 View 掩盖依赖。

## 10. 验收清单

### 10.1 元数据

- `/meta` 的公开语义目录恰好出现三个 View 和七个基础 Cube。
- `role_label_parent` 不进入公开目录。
- 三个 View 不使用 `includes: "*"`。
- 受治理 ID 成员保持 string 类型。
- 技术成员存在于 `/meta` 时必须保持 `public: false`，不能进入检索索引。
- `meta.join_edges` 与已验证的 Cube 关系完全一致。
- 强制行过滤引用的成员全部由对应 View 暴露。

### 10.2 查询结果

- 监测设备计数与数据库中“启用传感器关联启用设备”的对照一致；当前快照为 384。
- 单因素报警计数与数据库中“来源类型1且报警、传感器、设备均启用”的对照一致；当前快照为 1198。
- 多因素预警计数与数据库中“来源类型3且事件、配置均启用”的对照一致；当前快照为 84。
- 三个 View 的 ID 明细无非预期丢失或重复。
- 单标签和多标签筛选分别验证，并记录结果采用关系粒度还是传感器去重粒度。
- 报警事件选择设备、传感器字段后，事件计数不发生扇出。
- 多因素事件选择配置字段后，事件计数不发生扇出。
- 基础报警事件总数与按设备、按配置分组后的计数总和一致；当前快照均为 1504。
- 传感器标签唯一关系计数为 3882，源关系记录数为 6003；按标签分组后两项总和保持不变。
- `base_device_x_value.sensor_count` 当前快照为 438，证明基础 Cube 没有静默继承业务 View 的启用过滤。
- 至少验证一条跨 Cube 设备报警查询和一条跨 Cube 配置指标查询。
- 存在等长多路径的 Cube 组合在 `/load` 前返回 `semantic_model_gap`。

### 10.3 回归同步

- 同步 `cube/contract.py` 的公开 View、公开 Cube、Join Edge、隐藏成员和 string 成员集合。
- 同步目录选择测试中的真实 `/meta` 固件、四路 Top-K、成员反查和覆盖率断言。
- 清理或迁移场景内旧 View 名称引用。
- 运行 YAML 解析、Cube `/meta` 校验、代表性 `/load`、场景测试、静态检查和 Benchmark。

## 11. 需要业务人员最终确认的事项

1. `source_type = 1` 是否正式定义为单因素传感器报警，`source_type = 3` 是否正式定义为多因素预警。
2. 多标签筛选是“任意命中”还是“同时具备全部标签”；任意命中时明细是否要求每个传感器只返回一行。
3. 报警列表展示历史最高等级、当前等级，还是二者都展示。
4. 多因素启用配置清单是否仍是高频查询；如果是，应单独评估配置粒度入口，而不是让事件 View 同时承担两种根粒度。
5. 是否需要默认时间窗口、排序、分页、误报排除或解除状态过滤。文档当前没有给出这些口径。
