import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const outputDir = 'C:/Users/华硕/.codex/visualizations/2026/08/12/019ff3d9-fa3f-7192-8a6f-c0b9287a1167/outputs/秋招清单';
const verified = '2026-08-12';

const rows = [
['综合互联网','阿里巴巴','电商/云计算/本地生活','技术、产品、运营、设计、职能','已开放（2027届）','毕业时间：2026年11月—2027年10月','杭州/北京/上海等','https://campus-talent.alibaba.com/campus/gov','A'],
['综合互联网','百度','搜索/AI/云计算/自动驾驶','技术、产品、政企、销售、综合','已开放（2027届）','面向2026-09-01至2027-08-31毕业生','北京/上海/深圳等','https://talent.baidu.com/jobs/list?projectType=1','A'],
['综合互联网','字节跳动','内容平台/电商/企业服务/AI','研发、算法、产品、运营、设计','部分专项已开放（2027届）','前沿技术人才校招面向2027届及以后；常规秋招以官网筛选为准','北京/上海/深圳/杭州等','https://jobs.bytedance.com/campus','A'],
['综合互联网','腾讯','社交/游戏/云计算/内容','技术、产品、设计、市场、职能','官网入口开放，需筛选2027届','岗位批次与截止时间以官网为准','深圳/北京/上海/广州等','https://join.qq.com/','A'],
['综合互联网','美团','本地生活/零售/出行','技术、产品、商业分析、运营、职能','官网入口开放，需筛选2027届','建议尽早投递并留意补录','北京/上海/深圳/成都等','https://zhaopin.meituan.com/web/campus','A'],
['综合互联网','京东','电商/物流/科技/零售','技术、产品、供应链、运营、职能','官网入口开放，需筛选2027届','在校招项目中筛选应届生岗位','北京/上海/深圳/宿迁等','https://campus.jd.com/','A'],
['综合互联网','快手','短视频/直播/电商','技术、算法、产品、运营、设计','官网入口开放，需筛选2027届','岗位随业务动态更新','北京/杭州/深圳等','https://zhaopin.kuaishou.cn/recruit/e/#/campus','A'],
['综合互联网','拼多多','电商/跨境电商','技术、产品、运营、供应链','官网入口开放，需筛选2027届','关注校招岗位毕业时间要求','上海/广州/深圳等','https://careers.pinduoduo.com/campus/grad','A'],
['综合互联网','滴滴','出行/自动驾驶/国际化','技术、算法、产品、运营','官网入口开放，需筛选2027届','在校园招聘栏目查询最新批次','北京/上海/杭州等','https://talent.didiglobal.com/campus','B'],
['综合互联网','网易','游戏/音乐/内容/教育','技术、产品、运营、美术、策划','官网入口开放，需筛选2027届','集团与游戏岗位可能分批开放','杭州/广州/上海/北京等','https://campus.163.com/','A'],
['综合互联网','小红书','内容社区/电商','研发、算法、产品、运营、设计','官网入口开放，需筛选2027届','搜索校园招聘或应届生标签','上海/北京/武汉等','https://job.xiaohongshu.com/campus','B'],
['综合互联网','哔哩哔哩','视频/内容社区/游戏','技术、产品、运营、内容、设计','官网入口开放，需筛选2027届','岗位动态更新','上海/北京等','https://jobs.bilibili.com/campus','B'],
['综合互联网','携程集团','在线旅游/商旅','技术、产品、运营、业务、职能','官网入口开放，需筛选2027届','关注全球校招和国内校招批次','上海/北京/成都等','https://careers.ctrip.com/campus-recruitment','B'],
['综合互联网','同程旅行','在线旅游','研发、产品、运营、市场','官网入口开放，需筛选2027届','以官网最新岗位为准','苏州/北京/上海等','https://job.ly.com/campus','B'],
['综合互联网','得物App','潮流电商/社区','技术、产品、运营、供应链','官网入口开放，需筛选2027届','可同时关注实习转正机会','上海/杭州等','https://campus.dewu.com/','B'],
['综合互联网','货拉拉','同城货运/物流科技','技术、产品、运营、商业','官网入口开放，需筛选2027届','岗位和城市实时变化','深圳/香港/广州等','https://jobs.huolala.cn/#/campus','B'],
['硬件与智能终端','华为','ICT/云/终端/芯片','软件、硬件、算法、产品、销售','部分项目已开放（2027届）','2027届顶尖AI专项/实习已可见；秋招正式岗以官网为准','深圳/北京/上海/东莞等','https://career.huawei.com/cn/campus-recruitment','A'],
['硬件与智能终端','小米集团','手机/IoT/汽车/互联网服务','软件、硬件、算法、产品、运营','官网入口开放，需筛选2027届','在校园招聘栏目筛选毕业届次','北京/南京/武汉/深圳等','https://hr.xiaomi.com/campus','A'],
['硬件与智能终端','OPPO','智能终端/IoT','软件、硬件、算法、产品、设计','已开放（2027届）','官网岗位列表已标注2027届应届生校园招聘','深圳/东莞/成都/西安等','https://careers.oppo.com/campus/post','A'],
['硬件与智能终端','vivo','智能终端/影像/AI','软件、硬件、算法、产品、设计','官网入口开放，需筛选2027届','关注秋招正式岗位上线','东莞/深圳/杭州/南京等','https://hr.vivo.com/wt/vivo/web/index/CompvivoPageCampus','A'],
['硬件与智能终端','荣耀','智能终端/IoT','软件、硬件、算法、产品、营销','官网入口开放，需筛选2027届','按校园招聘项目查询','深圳/北京/西安/南京等','https://career.hihonor.com/SU60eea919bef57c1023f6fe78/pb/campus.html','B'],
['硬件与智能终端','联想集团','PC/服务器/智能设备','软件、硬件、算法、产品、销售','官网入口开放，需筛选2027届','选择校园招聘/Graduate岗位','北京/深圳/上海/武汉等','https://talent.lenovo.com.cn/','B'],
['AI与企业科技','科大讯飞','人工智能/大模型/教育科技','算法、研发、产品、解决方案','官网入口开放，需筛选2027届','技术岗较集中，留意提前批','合肥/北京/上海/广州等','https://campus.iflytek.com/','A'],
['AI与企业科技','大疆创新','无人机/机器人/影像','算法、软件、硬件、机械、产品','官网入口开放，需筛选2027届','校园招聘与创新项目分批开放','深圳/上海/西安等','https://we.dji.com/zh-CN/campus','A'],
['AI与企业科技','商汤科技','人工智能/计算机视觉','算法、研发、产品、解决方案','官网入口开放，需筛选2027届','关注研究和工程岗位','上海/北京/深圳/杭州等','https://joinus.sensetime.com/campus','B'],
['AI与企业科技','寒武纪','AI芯片','芯片设计、系统软件、算法、测试','官网入口开放，需筛选2027届','适合芯片/计算机体系结构方向','北京/上海/合肥/苏州等','https://career.cambricon.com/campus','B'],
['AI与企业科技','地平线机器人','智能驾驶芯片/算法','算法、软件、芯片、工具链','官网入口开放，需筛选2027届','关注校招和实习转正岗位','北京/上海/南京/深圳等','https://horizon.jobs.feishu.cn/campus','B'],
['AI与企业科技','金山办公','办公软件/云协作/AI','研发、算法、产品、运营、设计','官网入口开放，需筛选2027届','在校园招聘频道查询','北京/武汉/珠海/广州等','https://campus.wps.cn/','B'],
['AI与企业科技','深信服','网络安全/云计算','研发、测试、产品、售前、销售','官网入口开放，需筛选2027届','技术及客户成功类岗位较多','深圳/长沙/南京/全国等','https://hr.sangfor.com/campus','B'],
['AI与企业科技','海康威视','智能物联/视觉AI','研发、算法、硬件、产品、营销','官网入口开放，需筛选2027届','筛选校园招聘岗位','杭州/武汉/西安/成都等','https://campushr.hikvision.com/','B'],
['游戏与数字内容','米哈游','游戏/动画/内容','程序、策划、美术、运营、发行','已开放（2027届）','官网明确面向2026年9月—2027年8月毕业生','上海/北京/新加坡等','https://join.mihoyo.com/','A'],
['游戏与数字内容','莉莉丝游戏','游戏研发/发行','程序、策划、美术、发行、运营','官网入口开放，需筛选2027届','岗位滚动更新','上海/深圳等','https://app.mokahr.com/campus-recruitment/lilith/38897','B'],
['游戏与数字内容','鹰角网络','游戏研发/发行','程序、策划、美术、运营','官网入口开放，需筛选2027届','关注秋招项目页与岗位标签','上海等','https://campus.hypergryph.com/','B'],
['游戏与数字内容','叠纸游戏','游戏/互动娱乐','程序、策划、美术、运营','官网入口开放，需筛选2027届','美术和内容岗位较丰富','苏州/上海等','https://papegames.jobs.feishu.cn/campus','B'],
['游戏与数字内容','完美世界','游戏/影视','程序、策划、美术、运营','官网入口开放，需筛选2027届','选择校园招聘分类','北京/上海等','https://recruit.games.wanmei.com/campus','B'],
['金融科技与平台','蚂蚁集团','支付/金融科技/大模型','技术、算法、产品、风险、运营','官网入口开放，需筛选2027届','与阿里集团校招入口相互独立','杭州/上海/北京等','https://talent.antgroup.com/campus','A'],
['金融科技与平台','微众银行','数字银行/金融科技','技术、数据、产品、风险','官网入口开放，需筛选2027届','关注金融科技校招批次','深圳等','https://webank.jobs.feishu.cn/campus','B'],
['智能汽车','蔚来','智能汽车/智能座舱','软件、算法、硬件、产品、运营','官网入口开放，需筛选2027届','校园招聘岗位分批上线','上海/合肥/北京等','https://nio.jobs.feishu.cn/campus','A'],
['智能汽车','理想汽车','智能汽车/自动驾驶','算法、软件、硬件、产品、制造','官网入口开放，需筛选2027届','搜索“校园招聘/应届生”','北京/上海/常州等','https://www.lixiang.com/employ/campus.html','A'],
['智能汽车','小鹏汽车','智能汽车/自动驾驶','算法、软件、硬件、产品、制造','已启动（2027届项目）','2027届探索者计划已启动，具体岗位以官方入口为准','广州/深圳/上海/北京等','https://xiaopeng.jobs.feishu.cn/campus','A'],
['智能汽车','比亚迪','新能源汽车/电子/半导体','研发、软件、算法、制造、职能','官网入口开放，需筛选2027届','校招规模大，岗位与城市较多','深圳/西安/合肥/长沙等','https://job.byd.com/campus','A'],
['智能汽车','极氪','智能电动汽车','软件、算法、电子电气、产品','官网入口开放，需筛选2027届','关注智能驾驶和座舱岗位','宁波/杭州/上海等','https://zeekr.jobs.feishu.cn/campus','B'],
['云与基础设施','阿里云','云计算/AI/企业服务','研发、算法、产品、解决方案','部分项目已开放（2027届）','2027届管培/实习等项目可见','杭州/北京/上海/深圳等','https://careers.aliyun.com/campus/home','A'],
['云与基础设施','腾讯云','云计算/企业服务','研发、产品、解决方案、销售','随腾讯校招开放','通过腾讯校招官网筛选腾讯云相关岗位','深圳/北京/上海等','https://join.qq.com/','B'],
['云与基础设施','金山云','云计算/企业服务','研发、运维、产品、解决方案','官网入口开放，需筛选2027届','以校园招聘岗位实时列表为准','北京/武汉/上海等','https://app.mokahr.com/campus-recruitment/ksyun/37739','C'],
['大模型与Agent','MiniMax','基础大模型/多模态/Agent','模型算法、Agent、Infra、产品','已开放（2027届）','官网明确开放2027届校园招聘','上海/北京/杭州/海外','https://www.minimax.io/careers','A'],
['大模型与Agent','DeepSeek（深度求索）','基础大模型/推理/Agent','研究、训练推理系统、Agent Infra、产品','官方大量职位在招','官网有Agent后端、Agent Harness、Agent Infra、研究员等职位；逐岗确认应届要求','北京/杭州/乌兰察布','https://talent.deepseek.com/','A'],
['大模型与Agent','阶跃星辰 StepFun','基础大模型/多模态/Agent','大模型、多模态、Agent、语音、Infra','已开放（应届校招及实习）','StepStar顶尖人才计划已启动，截止信息以官网为准','上海/北京等','https://app.mokahr.com/campus-recruitment/step/141903#/jobs','A'],
['大模型与Agent','月之暗面 Moonshot AI','基础大模型/Kimi/Agent','模型算法、系统、产品、工程','校园招聘入口开放','官方飞书校园招聘入口，逐岗确认毕业届次','北京/杭州等','https://moonshot.jobs.feishu.cn/campus','A'],
['大模型与Agent','智谱 Z.ai','基础大模型/Agent/MaaS','算法、研发、产品、解决方案','校园招聘入口开放','官网提供校园招聘入口','北京/上海/杭州等','https://www.zhipuai.cn/zh/joinus','A'],
['大模型与Agent','百川智能','基础大模型/医疗AI','算法、研发、产品、AI应用','官方招聘入口开放','逐岗确认是否接受应届生','北京等','https://www.baichuan-ai.com/home#join','B'],
['大模型与Agent','零一万物 01.AI','基础大模型/AI应用','模型、Infra、应用研发、产品','官方招聘入口开放','逐岗确认是否接受应届生','北京/杭州等','https://www.lingyiwanwu.com/join-us','B'],
['大模型与Agent','面壁智能 ModelBest','端侧大模型/高效推理','算法、训练推理、研发、产品','官方招聘入口开放','适合端侧模型、模型压缩与推理方向','北京等','https://modelbest.jobs.feishu.cn/','B'],
['AI基础设施','硅基流动 SiliconFlow','AI Infra/模型云服务','推理优化、平台、后端、产品','官方招聘入口开放','逐岗确认校招或实习资格','北京/杭州等','https://siliconflow.jobs.feishu.cn/','A'],
['生成式AI应用','生数科技','多模态/视频生成','多模态算法、研发、产品、设计','官方招聘入口开放','逐岗确认校招或实习资格','北京/上海等','https://shengshu.jobs.feishu.cn/','B'],
['生成式AI应用','LiblibAI','AI图像/创作者平台','算法、研发、产品、运营','官方招聘入口开放','适合生成式AI应用与社区产品方向','北京等','https://liblib.jobs.feishu.cn/','B'],
['AI Agent应用','Manus','通用Agent/AI应用','Agent工程、模型、产品、增长','官方招聘入口开放','国际化团队，岗位多为社招；应届生需逐岗确认','北京/深圳/新加坡等','https://manus.im/careers','B'],
['AI Agent基础设施','Dify','Agent平台/开源AI应用开发','后端、前端、Agent、产品、开发者关系','官方招聘入口开放','全球化创业团队，逐岗确认应届资格','杭州/远程等','https://dify.ai/careers','B'],
['智能硬件与机器人','拓竹科技 Bambu Lab','3D打印/机器人/智能制造','软件、嵌入式、算法、机械、硬件','官方校招入口开放','官网与招聘平台可查看校招岗位，逐岗确认2027届','深圳/上海等','https://bambulab.com/en/career','A'],
['具身智能','宇树科技 Unitree','人形机器人/四足机器人','强化学习、控制、感知、嵌入式、机械','已开放（2027届）','2027届校园招聘已启动','杭州等','https://www.unitree.com/cn/position/','A'],
['具身智能','银河通用机器人 Galbot','具身大模型/人形机器人','具身算法、控制、感知、软件、硬件','已开放（2027届）','2027秋季校园招聘“具身领航者人才计划”已启动','北京/深圳/苏州/香港','https://galbot.jobs.feishu.cn/campus','A'],
['具身智能','智元机器人 AgiBot','人形机器人/具身智能','算法、控制、软件、硬件、机械','官方招聘入口开放','此前已开放2027届实习，正式秋招以官网为准','上海等','https://zhiyuan.jobs.feishu.cn/','A'],
['具身智能','逐际动力 LimX Dynamics','通用机器人/运动控制','强化学习、控制、感知、嵌入式、机械','官方招聘渠道开放','官网公布招聘邮箱，逐岗确认应届资格','深圳等','https://limx.cn/zh/careers','B'],
['具身智能','傅利叶智能','人形机器人/康复机器人','算法、控制、软件、硬件、机械','官方招聘入口开放','逐岗确认校招或实习资格','上海/新加坡等','https://fourierintelligence.com/career','B'],
['具身智能','星猿哲 XYZ Robotics','机器人视觉/智能物流','机器人算法、视觉、软件、控制','校园招聘入口开放','官网设有校园招聘入口','上海/深圳等','https://www.xyzrobotics.com.cn/careers','B'],
['具身智能','星动纪元 RobotEra','人形机器人/具身智能','强化学习、控制、感知、软件','官方招聘入口开放','逐岗确认校招或实习资格','北京等','https://robotera.jobs.feishu.cn/','B'],
['智能驾驶AI','Momenta','自动驾驶/大模型','感知、规划、仿真、数据、平台','校园招聘入口开放','关注2027届校招与实习转正岗位','苏州/北京/上海等','https://momenta.jobs.feishu.cn/campus','A'],
['智能驾驶AI','小马智行 Pony.ai','自动驾驶/Robotaxi','算法、系统、软件、硬件','官方招聘入口开放','逐岗确认校园招聘和毕业届次','北京/广州/深圳/上海等','https://www.pony.ai/job.html','B'],
['智能驾驶AI','文远知行 WeRide','自动驾驶/Robotaxi','算法、系统、软件、测试','官方招聘入口开放','逐岗确认校园招聘和毕业届次','广州/北京/上海/海外','https://www.weride.ai/jobs','B'],
['智能驾驶AI','元戎启行 DeepRoute.ai','自动驾驶/端到端模型','算法、平台、软件、数据','官方招聘入口开放','逐岗确认校招或实习资格','深圳/北京等','https://www.deeproute.ai/jobs','B'],
['AI芯片','摩尔线程','GPU/AI算力','芯片、编译器、系统软件、算法','校园招聘入口开放','适合GPU、编译器与高性能计算方向','北京/上海/成都等','https://career.mthreads.com/campus','A'],
['AI芯片','壁仞科技','GPU/AI芯片','芯片设计、编译器、系统、测试','校园招聘入口开放','逐岗确认2027届岗位','上海/北京等','https://career.birentech.com/campus','B'],
['AI芯片','燧原科技','AI芯片/算力平台','芯片、编译器、系统软件、算法','校园招聘入口开放','逐岗确认2027届岗位','上海/北京等','https://enflame.jobs.feishu.cn/campus','B'],
['AI芯片','黑芝麻智能','智能汽车芯片','芯片设计、工具链、算法、软件','校园招聘入口开放','关注自动驾驶芯片和工具链岗位','武汉/上海/深圳等','https://www.bst.ai/cn/campus','B'],
['央企·通信与数字科技','中国移动集团','通信/云计算/大数据/AI','软件、算法、云计算、网络安全、产品','官方招聘平台开放','集团及各专业公司岗位统一检索；2027届批次以公告为准','全国','https://job.10086.cn/','A'],
['央企·通信与数字科技','中国移动研究院','6G/大模型/网络智能/云计算','算法、研究、软件、系统、网络安全','随中国移动校招开放','在集团招聘平台筛选“研究院”','北京/上海/苏州等','https://job.10086.cn/','A'],
['央企·通信与数字科技','中移互联网有限公司','互联网产品/通信应用/AI','研发、算法、数据、产品、运营','随中国移动校招开放','在集团招聘平台筛选公司名称','广州/深圳等','https://job.10086.cn/','B'],
['央企·通信与数字科技','中移物联网有限公司','物联网/OneNET/智能硬件','研发、平台、嵌入式、算法、产品','随中国移动校招开放','在集团招聘平台筛选公司名称','重庆/成都/南京等','https://job.10086.cn/','B'],
['央企·通信与数字科技','咪咕公司','数字内容/视频/云游戏/AI','研发、算法、产品、运营、内容','随中国移动校招开放','适合互联网产品、音视频和推荐算法方向','北京/上海/南京/厦门等','https://job.10086.cn/','B'],
['央企·通信与数字科技','中国电信集团','通信/云计算/AI/安全','软件、算法、云计算、数据、安全','校园招聘岗位陆续上线','官方平台提示校招岗位陆续更新','全国','https://job.chinatelecom.com.cn/wt/TELE/web/index','A'],
['央企·通信与数字科技','中国电信人工智能研究院 TeleAI','大模型/AIGC/具身智能/AI Infra','大模型、算法、系统、数据、安全','已开放（2027人才计划）','TeleAI Top Talent 2027人才计划已启动','北京/上海等','https://job.chinatelecom.com.cn/wt/TELE/web/index','A'],
['央企·通信与数字科技','天翼云科技有限公司','云计算/大数据/AI','云平台、分布式系统、算法、安全、产品','随中国电信校招开放','在中国电信招聘平台筛选“天翼云”','北京/上海/杭州/广州等','https://job.chinatelecom.com.cn/wt/TELE/web/index','A'],
['央企·通信与数字科技','中国电信云计算研究院','云计算/数据库/分布式系统/AI','系统研究、数据库、网络、AI Infra','官方校招入口开放','在中国电信招聘平台筛选“云计算研究院”','北京/上海等','https://job.chinatelecom.com.cn/wt/TELE/web/index','A'],
['央企·通信与数字科技','中国联通集团','通信/云计算/大数据/AI','研发、算法、数据、网络安全、产品','官方招聘平台开放','2027届批次和单位以官网公告为准','全国','https://zhaopin.chinaunicom.cn/','A'],
['央企·通信与数字科技','联通数字科技有限公司','云计算/大数据/物联网/安全','软件、算法、云平台、解决方案','随中国联通校招开放','在集团招聘平台筛选“联通数科”','北京/全国','https://zhaopin.chinaunicom.cn/','A'],
['央企·通信与数字科技','中国联通软件研究院','通信软件/AI/大数据/安全','软件研发、算法、数据平台、安全','随中国联通校招开放','在集团招聘平台筛选“软件研究院”','北京/济南/哈尔滨等','https://zhaopin.chinaunicom.cn/','A'],
['央企·能源数字化','国家电网有限公司','能源互联网/电力数字化','软件、通信、数据、网络安全、AI','2027届正式公告待发布','预计按统一批次开放，以唯一官方平台为准','全国','https://zhaopin.sgcc.com.cn/','A'],
['央企·能源数字化','国网信息通信产业集团','能源互联网/云网/信创/AI','软件、算法、云平台、数据、安全','随国家电网统一招聘开放','在国网平台筛选“国网信通产业集团”','北京/南京/武汉等','https://zhaopin.sgcc.com.cn/','A'],
['央企·能源数字化','国网数字科技控股有限公司','能源数字化/区块链/大数据','软件、数据、算法、产品、解决方案','随国家电网统一招聘开放','在国网平台筛选“国网数科”','北京/雄安等','https://zhaopin.sgcc.com.cn/','A'],
['央企·能源数字化','国家电网信息通信中心','数字基础设施/数据/网络安全','平台研发、通信、数据治理、AI、安全','随国家电网统一招聘开放','总部直属数字化单位，竞争较高','北京等','https://zhaopin.sgcc.com.cn/','A'],
['央企·能源数字化','中国南方电网','数字电网/云计算/调度自动化','软件、通信、数据、AI、网络安全','官方招聘平台开放','2027届批次以官网公告为准','广东/广西/云南/贵州/海南','https://zhaopin.csg.cn/','A'],
['央企·电子信息','中国电子科技集团 CETC','电子信息/软件/网络安全/AI','算法、软件、系统、芯片、网络安全','官方招聘信息持续发布','研究所众多，需按城市和单位分别投递','全国','https://www.cetc.com.cn/zgdk/1593022/1592495/index.html','A'],
['央企·电子信息','中国电科智能科技研究院','人工智能/无人系统/云数智','AI研究、算法、软件、智能系统','官方招聘入口开放','中国电科集团层面的人工智能研究单位','北京等','https://www.cetc.com.cn/zgdk/1593022/1592495/index.html','A'],
['央企·电子信息','中电科数字科技集团','行业数字化/云计算/软件','软件、数据、云平台、网络安全','校园招聘入口开放','官网提供校园招聘信息','上海/北京等','https://www.shecc.com/job.html','B'],
['央企·电子信息','中电科金仓','国产数据库/基础软件','数据库内核、研发、测试、技术支持','官方招聘入口开放','适合数据库与基础软件方向','北京/武汉/成都等','https://www.kingbase.com.cn/about/recruitment.html','B'],
['央企·电子信息','中国电子信息产业集团 CEC','计算产业/信创/网络安全','软件、操作系统、数据库、安全、芯片','官方招聘信息持续发布','重点关注中国软件、中国长城、麒麟软件等成员单位','全国','https://www.cec.com.cn/rczp/index.html','A'],
['央企·航空航天','中国航天科技集团','航天软件/仿真/人工智能/信息化','软件、算法、控制、仿真、数据','官方校园招聘平台开放','下属院所和单位分别发布岗位','北京/上海/西安/成都等','https://www.spacetalent.com.cn/','A'],
['央企·航空航天','中国航天科工集团','航天防务/工业互联网/网络安全','软件、算法、信息安全、控制、仿真','官方招聘平台开放','关注航天云网及各研究院所','北京/武汉/南京/成都等','https://zhaopin.casic.cn/','A'],
['央企·航空航天','中国航空工业集团 AVIC','航空电子/软件/智能制造','软件、算法、控制、仿真、信息化','官方招聘平台开放','下属研究所与制造单位岗位差异较大','全国','https://job.avic.com/','A'],
['央企·高端制造','中国商飞 COMAC','民机/航电/仿真/工业软件','软件、算法、仿真、数据、信息化','官方招聘平台开放','关注设计研究院与信息化单位','上海/北京/成都等','https://zhaopin.comac.cc/','A'],
['央企·轨道交通','中国中车 CRRC','轨道交通/工业互联网/智能制造','软件、算法、控制、数据、信息化','官方招聘信息持续发布','下属研究所和子公司分别招聘','株洲/青岛/北京/上海等','https://www.crrcgc.cc/rczp','B'],
['央企·物流科技','中国邮政集团','物流科技/电商/金融科技','软件、数据、平台、网络安全、产品','联合校园招聘按年度开放','重点筛选信息科技、软件开发中心、数据中心','全国','https://www.chinapost.com.cn/html1/category/181312/8126-1.htm','B'],
['国有金融科技','中国工商银行科技条线','金融科技/手机银行/数据平台','前后端、数据、AI、安全、测试','校园招聘按年度开放','重点筛选科技菁英、业务研发中心、软件开发中心','北京/上海/广州/杭州/西安等','https://job.icbc.com.cn/','A'],
['国有金融科技','中国建设银行科技条线','金融科技/云平台/数据智能','软件、数据、AI、安全、产品','校园招聘按年度开放','重点关注总行直属中心和建信金融科技','北京/上海/武汉/成都等','https://job.ccb.com/','A'],
['国有金融科技','中国银行信息科技条线','金融科技/跨境支付/数据平台','软件、数据、AI、安全、运维','校园招聘按年度开放','重点筛选信息科技运营中心、软件中心','北京/上海/西安/合肥等','https://www.boc.cn/aboutboc/bi4/','A'],
['国有金融科技','中国农业银行科技条线','金融科技/数据平台/云计算','软件、数据、AI、安全、测试','校园招聘按年度开放','重点筛选研发中心、数据中心和科技岗','北京/广州/上海/天津等','https://career.abchina.com/','A'],
['国有金融科技','交通银行金融科技条线','金融科技/手机银行/数据智能','软件、数据、AI、安全、产品','校园招聘按年度开放','重点筛选金融科技部门和软件开发岗位','上海/北京/合肥等','https://job.bankcomm.com/','B'],
['国有金融科技','邮储银行信息科技条线','金融科技/数字银行/数据平台','软件、数据、AI、安全、运维','校园招聘按年度开放','与中国邮政集团部分岗位入口不同','北京/合肥/成都等','https://www.psbc.com/cn/gyyc/rczp/','B'],
['国有金融科技','中国银联','支付网络/金融科技/数据智能','软件、算法、数据、安全、产品','校园招聘按年度开放','适合支付系统、风控和数据方向','上海/北京等','https://career.unionpay.com/','A'],
['国有金融科技','中金公司 CICC 技术岗','证券科技/量化/交易系统','软件、数据、量化、AI、安全','校园招聘入口开放','金融科技岗位薪酬与竞争均较高','北京/上海/深圳等','https://careers.cicc.com.cn/','A']
];

const wb = Workbook.create();
const sheet = wb.worksheets.add('秋招投递清单');
const guide = wb.worksheets.add('使用说明');
sheet.showGridLines = false; guide.showGridLines = false;

sheet.getRange('A1:F1').merge();
sheet.getRange('A1').values = [['2027届互联网、AI及央国企数字岗位秋招清单']];
sheet.getRange('A1:F1').format = {fill:'#173F5F',font:{bold:true,color:'#FFFFFF',size:18},horizontalAlignment:'center',verticalAlignment:'center'};
sheet.getRange('A2:F2').merge();
sheet.getRange('A2').values = [[`核验日期：${verified}｜优先投递 A 级；“官网入口开放”不等于所有岗位均面向 2027 届，请进入官网确认毕业时间和截止日期。`]];
sheet.getRange('A2:F2').format = {fill:'#EAF2F8',font:{color:'#234B65',italic:true},wrapText:true};

const headers=['公司/企业','官方投递网址','建议优先级','我的状态','投递日期','备注'];
sheet.getRangeByIndexes(2,0,1,headers.length).values=[headers];
const data=rows.map(r=>[r[1],r[7],r[8],'未投递','', '']);
sheet.getRangeByIndexes(3,0,data.length,headers.length).values=data;

const end=3+rows.length;
sheet.getRange(`A3:F${end}`).format = {font:{name:'Microsoft YaHei',size:10},verticalAlignment:'center'};
sheet.getRange('A3:F3').format = {fill:'#20639B',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,borders:{preset:'outside',style:'thin',color:'#173F5F'}};
sheet.getRange(`A4:F${end}`).format.borders={insideHorizontal:{style:'thin',color:'#E5E7EB'}};
sheet.getRange(`B4:B${end}`).format={font:{color:'#0563C1',size:9,underline:true},wrapText:true};
sheet.getRange(`C4:E${end}`).format.horizontalAlignment='center';
sheet.getRange(`E4:E${end}`).format.numberFormat='yyyy-mm-dd';
sheet.freezePanes.freezeRows(3); sheet.freezePanes.freezeColumns(1);
sheet.getRange(`D4:D${end}`).dataValidation={rule:{type:'list',values:['未投递','准备中','已投递','笔试','面试','Offer','已结束']}};
sheet.getRange(`C4:C${end}`).conditionalFormats.add('containsText',{text:'A',format:{fill:'#DCFCE7',font:{bold:true,color:'#166534'}}});
sheet.getRange(`C4:C${end}`).conditionalFormats.add('containsText',{text:'B',format:{fill:'#FEF3C7',font:{color:'#92400E'}}});
sheet.getRange(`C4:C${end}`).conditionalFormats.add('containsText',{text:'C',format:{fill:'#F3F4F6',font:{color:'#4B5563'}}});
sheet.getRange(`D4:D${end}`).conditionalFormats.add('containsText',{text:'已投递',format:{fill:'#DBEAFE',font:{color:'#1D4ED8',bold:true}}});
sheet.getRange(`D4:D${end}`).conditionalFormats.add('containsText',{text:'Offer',format:{fill:'#DCFCE7',font:{color:'#166534',bold:true}}});
const table=sheet.tables.add(`A3:F${end}`,true,'RecruitmentTable'); table.style='TableStyleMedium2'; table.showFilterButton=true;
// 央国企及国有金融科技单位使用浅金色整行底色，方便与其他企业快速区分。
rows.forEach((r,i)=>{
  if (r[0].startsWith('央企') || r[0].startsWith('国有')) {
    sheet.getRange(`A${i+4}:F${i+4}`).format.fill='#FFF2CC';
  }
});
const widths=[19,60,14,15,16,38];
widths.forEach((w,i)=>sheet.getRangeByIndexes(0,i,end,1).format.columnWidth=w);
sheet.getRange('A1:F1').format.rowHeight=34; sheet.getRange('A2:F2').format.rowHeight=34; sheet.getRange(`A4:F${end}`).format.rowHeight=34;

guide.getRange('A1:F1').merge(); guide.getRange('A1').values=[['使用说明与投递策略']];
guide.getRange('A1:F1').format={fill:'#173F5F',font:{bold:true,color:'#FFFFFF',size:18},horizontalAlignment:'center'};
guide.getRange('A3:B9').values=[
['字段','说明'],['A/B/C 优先级','A：建议优先查看；B：作为扩展；C：补充关注'],['当前状态','仅“已开放（2027届）”表示检索时明确看到届次信息'],['官网入口开放','官方招聘网站可访问，但需自行筛选“2027届/应届生”'],['我的状态','提供下拉选项，可自行记录投递进度'],['链接使用','点击“一键投递”；旁边保留完整官方网址便于核验'],['安全提示','任何收费内推、收费笔试、私人账户转账均需警惕']];
guide.getRange('A11:F11').merge(); guide.getRange('A11').values=[['建议节奏']];
guide.getRange('A12:B16').values=[['阶段','建议动作'],['现在','优先完成 A 级公司官网注册、简历和作品集'],['每周 2 次','回查“需筛选2027届”的企业是否新增正式岗位'],['投递当天','记录岗位名称、岗位编号、城市和截止日期'],['笔试/面试后','在“备注”记录题型、面试轮次和待跟进事项']];
guide.getRange('D7:E9').values=[['颜色图例','企业性质'],['浅金色','央企/国企及国有金融科技'],['蓝白交替','互联网、AI及其他市场化企业']];
guide.getRange('D7:E7').format={fill:'#20639B',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center'};
guide.getRange('D8:E8').format.fill='#FFF2CC';
guide.getRange('D9:E9').format.fill='#D9EAF7';
guide.getRange('D3:F3').values=[['A级企业','央国企单位','总企业数']];
guide.getRange('D4').formulas=[[`=COUNTIF('秋招投递清单'!$C$4:$C$${end},"A")`]];
guide.getRange('E4').values=[[rows.filter(r=>r[0].startsWith('央企')||r[0].startsWith('国有')).length]];
guide.getRange('F4').formulas=[[`=COUNTA('秋招投递清单'!$A$4:$A$${end})`]];
guide.getRange('A3:B3').format={fill:'#20639B',font:{bold:true,color:'#FFFFFF'}};
guide.getRange('A11:F11').format={fill:'#D6EAF8',font:{bold:true,color:'#173F5F'}};
guide.getRange('A12:B12').format={fill:'#20639B',font:{bold:true,color:'#FFFFFF'}};
guide.getRange('D3:F3').format={fill:'#20639B',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true};
guide.getRange('D4:F4').format={fill:'#EAF2F8',font:{bold:true,color:'#173F5F',size:16},horizontalAlignment:'center'};
guide.getRange('A3:F16').format.borders={insideHorizontal:{style:'thin',color:'#E5E7EB'}};
guide.getRange('A1:F16').format={font:{name:'Microsoft YaHei'},verticalAlignment:'center'};
guide.getRange('A1:F1').format={fill:'#173F5F',font:{name:'Microsoft YaHei',bold:true,color:'#FFFFFF',size:18},horizontalAlignment:'center'};
['A','B','C','D','E','F'].forEach((c,i)=>guide.getRange(`${c}:${c}`).format.columnWidth=[18,54,4,20,20,18][i]);
guide.getRange('A3:F16').format.wrapText=true; guide.freezePanes.freezeRows(1);

await fs.mkdir(outputDir,{recursive:true});
const inspect=await wb.inspect({kind:'table',range:`秋招投递清单!A1:F10`,include:'values,formulas',tableMaxRows:10,tableMaxCols:6,maxChars:5000});
console.log(inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:100},summary:'formula scan'});
console.log(errors.ndjson);
for (const name of ['秋招投递清单','使用说明']) {
 const p=await wb.render({sheetName:name,autoCrop:'all',scale:1,format:'png'});
 await fs.writeFile(`${outputDir}/${name}.png`,new Uint8Array(await p.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(`${outputDir}/2027届秋招清单_精简状态列_110家.xlsx`);
console.log(`EXPORTED ${outputDir}/2027届秋招清单_精简状态列_110家.xlsx ROWS=${rows.length}`);
